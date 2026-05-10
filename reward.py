# %% [markdown]
# # Trains a reward model out of a dataset of human preferences.
# # Used to evaluate DPO.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np

# %%
# use if not installed already
# %pip install torch==2.6.0 #+cu124
# %pip install transformers==5.5.4 peft==0.19.1 matplotlib

# %%
# load tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B_sft_step_13000"

tokenizer = AutoTokenizer.from_pretrained("checkpoints/" + model_name) # load from local checkpoint
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/" + model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
print(f"Model Name: {model_name}")

# %%
print(model)

# %%
# defines a reward model. ignore lm_head, use reward_head directly on last hidden states


import torch.nn as nn

class QwenRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        # Keep only the transformer backbone
        self.model = base_model.model

        hidden_size = base_model.config.hidden_size  # 896

        # Scalar reward head
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, model_inputs):
        outputs = self.model(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
            labels=None,
            use_cache=False,
        )

        # last_hidden_state:
        # [batch_size, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state

        # Get final token hidden state for each sequence
        # Usually use the last non-padding token
        attention_mask = model_inputs.get("attention_mask", None)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[
                torch.arange(hidden_states.size(0)),
                seq_lengths
            ]
        else:
            last_hidden = hidden_states[:, -1]

        # reward shape: [batch_size, 1]
        reward = self.reward_head(last_hidden)

        # squeeze to [batch_size]
        return reward.squeeze(-1)

reward_model = QwenRewardModel(model).to(model.device)
# cast model to bf16
reward_model = reward_model.to(torch.bfloat16)
texts = [
    "Tell me a joke.",
    "How do I build a bomb?"
]

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(model.device)
print(batch)
rewards = reward_model(batch)

print(rewards)
print(rewards.shape)

# %%
# load dataset 
# the train and test splits contain everything for sft and dpo, so we can use them for both stages
train_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
test_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
# in sft stage, use 'message'
# in dpo stage, use 'chosen' and 'rejected'

print("Example Data:")
print("Train Data:")
print(train_ds[0])
print("Test Data:")
print(test_ds[0])

# %% [markdown]
# Plan: use full finetuning for SFT, then use LoRA for DPO, because the model size is 0.5B.
# At 0.5B, SFT may require a large representation shift. LoRA may be limiting.

# %%
train_ds[0]["messages"]
train_ds[0]["chosen"]
train_ds[0]["rejected"]

# %%
# set up torch device
if torch.cuda.is_available():
    device = torch.device( "cuda" )
elif torch.backends.mps.is_available():
    device = torch.device( "mps" )
else:
    device = torch.device( "cpu" )

# set up hyperparameters
EPOCHS = 2
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 1024
REWARD_CENTERING = 0.001 # penalize coefficient. Additional loss: reward_centering * (reward)^2
EVAL_INTERVAL = 1000  # validate every 1000 steps

print(f"Hyperparameters:\n \
        Epochs: {EPOCHS}\n \
        Learning Rate: {LEARNING_RATE}\n \
        Batch Size: {BATCH_SIZE}\n \
        Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}\n \
        Max Sequence Length: {MAX_LENGTH}\n \
        Reward Centering Coefficient: {REWARD_CENTERING}\n \
        Validation Frequency (steps): {EVAL_INTERVAL}\n \n")

# %%
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# %%
# model.save_pretrained(f"checkpoints/{model_name}_sft_epoch_0")
# model = AutoModelForCausalLM.from_pretrained(f"checkpoints/{model_name}_sft_epoch_0", device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa")

# %%
# define train helpers

def unbatch_chat_messages(batched_messages):
    batch_size = len(batched_messages[0]["content"])
    return [
        [
            {
                "role": message_group["role"][sample_index],
                "content": message_group["content"][sample_index],
            }
            for message_group in batched_messages
        ]
        for sample_index in range(batch_size)
    ]

def mask_non_assistant_response(model_inputs, tokenizer):
    # the next position after <|im_start|>assistant
    # <|im_start|>assistant token id is 151644 followed by 77091
    start_pos = (model_inputs["input_ids"] == 151644) & (model_inputs["input_ids"].roll(-1, dims=1) == 77091)
    start_pos = start_pos * torch.arange(model_inputs["input_ids"].shape[1], device=model_inputs["input_ids"].device)
    start_pos = start_pos.max(dim=1).values + 2  # add 2 to get to the position of the first token of the response  
    # compute labels, only compute loss on assistant response
    labels = model_inputs["input_ids"].clone()
    pos = torch.arange(model_inputs["input_ids"].shape[1], device=model_inputs["input_ids"].device).unsqueeze(0) # (1, seq_len)
    mask = pos < start_pos.unsqueeze(1)  # (batch_size, seq_len)
    labels[mask] = tokenizer.pad_token_id  # set non-response tokens to padding
    return labels

def prepare_inputs(messages, tokenizer, device):
    conversations = unbatch_chat_messages(messages)
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in conversations
    ]
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)
    return model_inputs


# %%
# setup full-finetuning training loop for SFT
from tqdm import tqdm
import torch.optim as optim

# set up optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS*len(train_dataloader), eta_min=1e-6)

train_loss_history = [] # for each step (batch)
eval_loss_history = [] # for each validation step
eval_loss_steps = [] # step numbers when validation was performed
epoch_end_steps = [] # to keep track of step number at the end of each epoch for plotting
eval_win_rate_history = [] # to record the rate at which the reward model prefers the chosen response over the rejected response in the eval set
best_eval_loss = float('inf')
global_step = 0

def plot_loss_curves():
    # plot training and eval loss curves
    train_loss_history_arr = np.array(train_loss_history)
    eval_loss_history_arr = np.array(eval_loss_history)
    eval_loss_steps_arr = np.array(eval_loss_steps)
    step_num_history_arr = np.arange(1, len(train_loss_history) + 1)
    
    # plot and save the training and validation loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step_num_history_arr, train_loss_history_arr, label='Training Loss', alpha=0.7)
    ax.plot(eval_loss_steps_arr, eval_loss_history_arr, label='Validation Loss', marker='o', markersize=4, alpha=0.7)
    ax.legend(loc='upper left')
    # plot centering penalties on a separate y axis
    ax2 = ax.twinx()
    ax2.plot(eval_loss_steps_arr, eval_win_rate_history, label='Eval Win Rate', color='orange', marker='x', markersize=4, alpha=0.7)
    # set y range to 0-1 for win rate
    ax2.set_ylim(0, 1)
    # set y label for the centering penalty axis
    ax2.set_ylabel('Eval Win Rate')
    ax2.legend(loc='upper right')
    # add vertical lines at epoch boundaries
    for step_idx in epoch_end_steps:
        ax.axvline(step_idx, color='gray', ls='--', lw=0.6, alpha=0.35)

    # add epoch ticks on the top axis
    if len(epoch_end_steps) > 0:
        epoch_end_steps_arr = np.array(epoch_end_steps)
        epoch_ids = np.arange(1, len(epoch_end_steps_arr) + 1)
        max_labels = 12
        stride = max(1, int(np.ceil(len(epoch_end_steps_arr) / max_labels)))
        top_ticks = epoch_end_steps_arr[::stride]
        top_labels = epoch_ids[::stride]
        top_ax = ax.secondary_xaxis('top')
        top_ax.set_xticks(top_ticks)
        top_ax.set_xticklabels(top_labels)
        top_ax.set_xlabel('Epoch')
    
    # add a horizontal line showing minimum val loss
    ax.axhline(y=best_eval_loss, color='black', linestyle='--', linewidth=1, alpha=0.2, label=f'Min Eval Loss: {best_eval_loss:.4g}')

    ax.set_xlabel('Step Number')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curves for {model_name.split("/")[-1]} Reward Training')
    ax.legend()
    ax.grid(False)
    # # change y to log scale
    # ax.set_yscale('log')
    # ax.set_ylim(0.0001, 0.01)
    fig.tight_layout()
    fig.savefig(f"reward_loss_curve.png")
    
for epoch in range(EPOCHS):
    # train
    pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{EPOCHS}")
    optimizer.zero_grad()
    accumulation_step = 0
    for batch in pbar:
        win_model_inputs = prepare_inputs(batch["chosen"], tokenizer, model.device)
        lose_model_inputs = prepare_inputs(batch["rejected"], tokenizer, model.device)
        
        win_rewards = reward_model(win_model_inputs)
        lose_rewards = reward_model(lose_model_inputs)
        
        # compute max-likelihood loss
        loss = -torch.log(
            torch.sigmoid(
                win_rewards - lose_rewards
            )
        )
        # here, loss has shape [B], each entry is the loss of i-th batch!
        # remember batch reduction! take mean of the loss.
        loss = torch.mean(loss)
        train_loss_history.append(loss.detach().item()) # record the loss before reward centering penalty for better visualization of training dynamics
        # add reward centering penalty
        centering_penalty = torch.mean(win_rewards**2 + lose_rewards**2)
        loss = loss + REWARD_CENTERING * centering_penalty
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        accumulation_step += 1
        if accumulation_step % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        
        # Step-based validation
        if global_step % EVAL_INTERVAL == 0:
            pbar_eval = tqdm(test_dataloader, desc=f"Eval at Step {global_step}", leave=False)
            eval_loss = 0
            eval_win_rate = 0
            eval_batch_count = 0
            kl_div = 0
            for batch_eval in pbar_eval:
                # compute DPO loss on the eval set
                win_model_inputs_eval = prepare_inputs(batch_eval["chosen"], tokenizer, model.device)
                lose_model_inputs_eval = prepare_inputs(batch_eval["rejected"], tokenizer, model.device)
                # forward passes
                with torch.no_grad():
                    win_rewards_eval = reward_model(win_model_inputs_eval)
                    lose_rewards_eval = reward_model(lose_model_inputs_eval)
                    loss = -torch.log(
                        torch.sigmoid(
                            win_rewards_eval - lose_rewards_eval
                        )
                    )
                    loss = torch.mean(loss)
                    eval_loss += loss.detach().item()
                    eval_batch_count += 1
                    
                    # compute win rate
                    win_rate = torch.mean((win_rewards_eval > lose_rewards_eval).float())
                    eval_win_rate += win_rate.detach().item()

                    # centering_penalty_eval = torch.mean(win_rewards_eval**2 + lose_rewards_eval**2)
                    
            eval_loss /= eval_batch_count
            eval_win_rate /= eval_batch_count
            eval_win_rate_history.append(eval_win_rate)
            eval_loss_history.append(eval_loss)
            eval_loss_steps.append(global_step)
            tqdm.write(f"Step {global_step}: Train Loss={train_loss_history[-1]:.4f}, Eval Loss={eval_loss:.4f}, Eval Win Rate={eval_win_rate:.6f}")
            
            # update best eval loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model.save_pretrained(f"checkpoints/{model_name}_reward_step_{global_step}")
                tokenizer.save_pretrained(f"checkpoints/{model_name}_reward_step_{global_step}")
                print(f"New best model saved with eval loss {best_eval_loss:.4f}")
            
            plot_loss_curves()

    if accumulation_step % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    epoch_end_steps.append(len(train_loss_history))
    




