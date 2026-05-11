# %% [markdown]
# # Performs DPO on a SFTed model

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
# set up torch device
if torch.cuda.is_available():
    device = torch.device( "cuda" )
elif torch.backends.mps.is_available():
    device = torch.device( "mps" )
else:
    device = torch.device( "cpu" )
    
# load tokenizer and model
from pathlib import Path
import torch.nn as nn

model_name = "Qwen/Qwen2.5-0.5B_sft_step_13000"
tokenizer = AutoTokenizer.from_pretrained("checkpoints/" + model_name)
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/" + model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
print(f"Model Name: {model_name}")



# %%
class QwenRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model.model
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, model_inputs):
        attention_mask = model_inputs["attention_mask"]
        # with left padding, and batch gen padding on the right, the sequence might look like
        # [eos, eos, eos, (useful content in the middle), eos eos eos eos]
        # and cuz we use eos as pad token
        # find the last valid (non-padding) position for each sequence
        idx = torch.where(attention_mask == 1, torch.arange(attention_mask.size(1)).to(self.model.device), -1).max(dim=1).values
        
        last_token_ids = model_inputs['input_ids'][torch.arange(attention_mask.size(0)), idx]
        print(last_token_ids)
        # ensure that the reward head sees the terminating id == 198, otherwise behavior is inconsistent with its training
        model_inputs['input_ids'][torch.arange(attention_mask.size(0)), idx] = 198
        if not torch.all(last_token_ids == model_inputs['input_ids'][torch.arange(attention_mask.size(0)), idx]):
            print("Warning: last token ids set to 198 for correct evaluation.")
            print(model_inputs['input_ids'][torch.arange(attention_mask.size(0)), idx])

        outputs = self.model(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
            labels=None,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        last_hidden = hidden_states[torch.arange(hidden_states.size(0)), idx]
        
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)

reward_model_name = "Qwen/Qwen2.5-0.5B_sft_step_13000_reward_step_7000"
reward_tokenizer = AutoTokenizer.from_pretrained("checkpoints/" + reward_model_name)
reward_base_model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/" + reward_model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
reward_model = QwenRewardModel(reward_base_model)
reward_head_path = Path("checkpoints") / reward_model_name / "reward_head.pt"
# separately load reward head
reward_model.reward_head.load_state_dict(torch.load(reward_head_path, map_location=next(reward_model.parameters()).device), strict=True)
reward_model = reward_model.to(torch.bfloat16).to(device)
reward_model.eval()
print(f"Reward Model Name: {reward_model_name}")

# %%
#test raw model
_prompts = ["Hi who are you?", "1+1=?"]
texts = [
    tokenizer.apply_chat_template([{"role": "user", "content": _prompt}], tokenize=False, add_generation_prompt=True, return_tensors="pt")
    for _prompt in _prompts
]

inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding_side="left",
    padding=True,
).to(device)

model.eval()
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
    )
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
print("Raw Model Generation:\n")
for c in generated_text:
    print(c) # you see long tails of eof tokens, because this is batch generation, they are "padding"



# %%
reward_model.eval()
reward_inputs = reward_tokenizer(generated_text, return_tensors="pt").to(next(reward_model.parameters()).device)
# manually set attention mask to ignore eof (id == 151643)
reward_inputs['attention_mask'][reward_inputs['input_ids'] == 151643] = 0
with torch.no_grad():
    rewards = reward_model(reward_inputs)
print(rewards)

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
# set up hyperparameters
EPOCHS = 2
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 1024
BETA = 0.05
LORA_RANK = 16
EVAL_INTERVAL = 3000  # validate every N steps
REWARD_EVAL_PROMPT_COUNT = 16 # use a fixed subset of test set prompt to calculate reward win rate on open-end generation

print(f"Hyperparameters:\n \
        Epochs: {EPOCHS}\n \
        Learning Rate: {LEARNING_RATE}\n \
        Batch Size: {BATCH_SIZE}\n \
        Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}\n \
        Max Sequence Length: {MAX_LENGTH}\n \
        Beta (DPO): {BETA}\n \
        LoRA Rank: {LORA_RANK}\n \
        Validation Frequency (steps): {EVAL_INTERVAL}\n \
        REWARD_EVAL_PROMPT_COUNT: {REWARD_EVAL_PROMPT_COUNT}\n \n")

# %%
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# %%
for batch in test_dataloader:
    prompts = batch["prompt"]
    processed_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
    inputs = tokenizer(processed_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(model.device)
    print(inputs)
    break

# %%
# prepare the model for lora
lora_config = LoraConfig(
    r=LORA_RANK, # the rank
    lora_alpha=16, # the scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# %%
# check out the injected lora layers!
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
print(f"Number of trainable parameters: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)}")
print(f"Number of trainable layers: {len(trainable)}")

# %%
# get reference model by no_grad and temporarily disabling LoRA
prompt = "Describe the property of sin and cos functions. List the properties one by one."
text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").to(model.device)
model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

model.eval()
policy_output = model(**model_inputs)

model.eval()
with torch.no_grad():
    with model.disable_adapter():
        ref_outputs = model(**model_inputs)

print("policy logits shape:", policy_output.logits.shape)
print("ref logits shape:", ref_outputs.logits.shape)
print(policy_output.logits)

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
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS*len(train_dataloader)/GRADIENT_ACCUMULATION_STEPS, eta_min=1e-6)

train_loss_history = [] # for each step (batch)
eval_loss_history = [] # for each validation step
eval_loss_steps = [] # step numbers when validation was performed
epoch_end_steps = [] # to keep track of step number at the end of each epoch for plotting
kl_div_history = [] # compare the output distribution of the current model and the reference model at each validation step
win_rate_history = [] # track the win rate of the current model against the reference model at each validation step
best_eval_loss = float('inf')
global_step = 0

def plot_loss_curves():
    # prepare arrays
    train_loss_history_arr = np.array(train_loss_history)
    eval_loss_history_arr = np.array(eval_loss_history)
    eval_loss_steps_arr = np.array(eval_loss_steps)
    step_num_history_arr = np.arange(1, len(train_loss_history) + 1)
    kl_div_history_arr = np.array(kl_div_history)
    win_rate_history_arr = np.array(win_rate_history)
    
    # create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # subplot 1: training and validation losses
    ax1.plot(step_num_history_arr, train_loss_history_arr, label='Training Loss', alpha=0.7)
    ax1.plot(eval_loss_steps_arr, eval_loss_history_arr, label='Validation Loss', marker='o', markersize=4, alpha=0.7)
    ax1.axhline(y=best_eval_loss, color='black', linestyle='--', linewidth=1, alpha=0.2, label=f'Min Eval Loss: {best_eval_loss:.4g}')
    
    # add vertical lines at epoch boundaries for both subplots
    for step_idx in epoch_end_steps:
        ax1.axvline(step_idx, color='gray', ls='--', lw=0.6, alpha=0.35)
        ax2.axvline(step_idx, color='gray', ls='--', lw=0.6, alpha=0.35)
    
    # add epoch ticks on top axis for subplot 1
    if len(epoch_end_steps) > 0:
        epoch_end_steps_arr = np.array(epoch_end_steps)
        epoch_ids = np.arange(1, len(epoch_end_steps_arr) + 1)
        max_labels = 12
        stride = max(1, int(np.ceil(len(epoch_end_steps_arr) / max_labels)))
        top_ticks = epoch_end_steps_arr[::stride]
        top_labels = epoch_ids[::stride]
        top_ax1 = ax1.secondary_xaxis('top')
        top_ax1.set_xticks(top_ticks)
        top_ax1.set_xticklabels(top_labels)
        top_ax1.set_xlabel('Epoch')
    
    ax1.set_xlabel('Step Number')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training & Validation Loss - {model_name.split("/")[-1]} DPO')
    ax1.legend(loc='upper right')
    ax1.grid(False)
    
    # subplot 2: win rate and KL divergence
    if len(win_rate_history_arr) > 0:
        ax2.plot(eval_loss_steps_arr, win_rate_history_arr, label='Win Rate', color='green', marker='o', markersize=4, alpha=0.7)
    if len(kl_div_history_arr) > 0:
        ax2_dup = ax2.twinx()  # create a secondary y-axis for KL divergence
        ax2_dup.plot(eval_loss_steps_arr, kl_div_history_arr, label='KL Divergence', color='purple', marker='s', markersize=4, alpha=0.7)
    
    # add epoch ticks on top axis for subplot 2
    if len(epoch_end_steps) > 0:
        top_ax2 = ax2.secondary_xaxis('top')
        top_ax2.set_xticks(top_ticks)
        top_ax2.set_xticklabels(top_labels)
        top_ax2.set_xlabel('Epoch')
    
    ax2.set_xlabel('Step Number')
    ax2.set_ylabel('Win Rate')
    ax2.set_ylim(0, 1)  # win rate is between 0 and 1
    ax2.set_title(f'Win Rate & KL Divergence - {model_name.split("/")[-1]} DPO')
    ax2.legend(loc='upper right')
    ax2.grid(False)
    ax2_dup.set_ylabel('KL Divergence')
    ax2_dup.legend(loc='upper left')
    ax2_dup.grid(False)

    fig.tight_layout()
    fig.savefig(f"dpo_loss_curve.png")
    
for epoch in range(EPOCHS):
    # train
    pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{EPOCHS}")
    optimizer.zero_grad()
    accumulation_step = 0
    for batch in pbar:
        model.train()
        win_model_inputs = prepare_inputs(batch["chosen"], tokenizer, model.device)
        win_labels = mask_non_assistant_response(win_model_inputs, tokenizer)
        lose_model_inputs = prepare_inputs(batch["rejected"], tokenizer, model.device)
        lose_labels = mask_non_assistant_response(lose_model_inputs, tokenizer)
        win_input_ids_shifted = torch.roll(win_model_inputs["input_ids"], shifts=-1, dims=1)
        lose_input_ids_shifted = torch.roll(lose_model_inputs["input_ids"], shifts=-1, dims=1)
        
        # forward passes
        lora_win_outputs = model(**win_model_inputs, labels=None, use_cache=False) # disable KV cache
        lora_lose_outputs = model(**lose_model_inputs, labels=None, use_cache=False) # disable KV cache
        lora_win_log_prob = torch.nn.functional.log_softmax(lora_win_outputs.logits, dim=-1)
        lora_lose_log_prob = torch.nn.functional.log_softmax(lora_lose_outputs.logits, dim=-1)
        # consider log probs of only token positions in training data
        lora_win_log_prob_gathered = torch.gather(lora_win_log_prob, dim=2, index=win_input_ids_shifted.unsqueeze(-1)).squeeze(-1)
        lora_lose_log_prob_gathered = torch.gather(lora_lose_log_prob, dim=2, index=lose_input_ids_shifted.unsqueeze(-1)).squeeze(-1)
        win_mask = torch.roll(win_labels != tokenizer.pad_token_id, shifts=-1, dims=1) # shift the mask to align with the log probs of the next tokens
        lose_mask = torch.roll(lose_labels != tokenizer.pad_token_id, shifts=-1, dims=1)
        lora_win_sentence_log_prob = (lora_win_log_prob_gathered * win_mask)[:, :-1].sum(dim=1)
        lora_lose_sentence_log_prob = (lora_lose_log_prob_gathered * lose_mask)[:, :-1].sum(dim=1)
        # get reference model outputs without gradient and adapter
        with torch.no_grad():
            with model.disable_adapter():
                model.eval() # make sure to set reference model to eval mode for stable evaluation!
                ref_win_outputs = model(**win_model_inputs, labels=None, use_cache=False)
                ref_lose_outputs = model(**lose_model_inputs, labels=None, use_cache=False)
                ref_win_log_prob = torch.nn.functional.log_softmax(ref_win_outputs.logits, dim=-1)
                ref_lose_log_prob = torch.nn.functional.log_softmax(ref_lose_outputs.logits, dim=-1)
                ref_win_log_prob_gathered = torch.gather(ref_win_log_prob, dim=2, index=win_input_ids_shifted.unsqueeze(-1)).squeeze(-1)
                ref_lose_log_prob_gathered = torch.gather(ref_lose_log_prob, dim=2, index=lose_input_ids_shifted.unsqueeze(-1)).squeeze(-1)
                # win_mask and lose_mask are the same for LoRA and reference since they are based on the input labels
                ref_win_sentence_log_prob = (ref_win_log_prob_gathered * win_mask)[:, :-1].sum(dim=1)
                ref_lose_sentence_log_prob = (ref_lose_log_prob_gathered * lose_mask)[:, :-1].sum(dim=1)
        model.train() # switch back to train mode for the policy model after getting reference outputs
        # compute DPO loss
        loss = -torch.log(
            torch.sigmoid(
                BETA * (
                    lora_win_sentence_log_prob - lora_lose_sentence_log_prob
                    - ref_win_sentence_log_prob + ref_lose_sentence_log_prob
                )
            )
        )
        # here, loss has shape [B], each entry is the loss of i-th batch!
        # remember batch reduction! take mean of the loss.
        loss = torch.mean(loss)
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        accumulation_step += 1
        if accumulation_step % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss_history.append(loss.detach().item())
        # global_step increment at the end of validation block to make sure first validation happens at step 0
        
        # Step-based validation
        if global_step % EVAL_INTERVAL == 0:
            model.eval()
            pbar_eval = tqdm(test_dataloader, desc=f"Eval at Step {global_step}", leave=False)
            eval_loss = 0
            eval_batch_count = 0
            kl_div = 0
            for batch_eval in pbar_eval:
                # compute DPO loss on the eval set
                win_model_inputs_eval = prepare_inputs(batch_eval["chosen"], tokenizer, model.device)
                win_labels_eval = mask_non_assistant_response(win_model_inputs_eval, tokenizer)
                lose_model_inputs_eval = prepare_inputs(batch_eval["rejected"], tokenizer, model.device)
                lose_labels_eval = mask_non_assistant_response(lose_model_inputs_eval, tokenizer)
                win_input_ids_shifted_eval = torch.roll(win_model_inputs_eval["input_ids"], shifts=-1, dims=1)
                lose_input_ids_shifted_eval = torch.roll(lose_model_inputs_eval["input_ids"], shifts=-1, dims=1)
                # forward passes
                with torch.no_grad():
                    model.eval() # make sure to set eval mode for stable evaluation!
                    lora_win_outputs_eval = model(**win_model_inputs_eval, labels=None, use_cache=False) # disable KV cache
                    lora_lose_outputs_eval = model(**lose_model_inputs_eval, labels=None, use_cache=False) # disable KV cache
                    lora_win_log_prob_eval = torch.nn.functional.log_softmax(lora_win_outputs_eval.logits, dim=-1)
                    lora_lose_log_prob_eval = torch.nn.functional.log_softmax(lora_lose_outputs_eval.logits, dim=-1)
                    lora_win_log_prob_gathered_eval = torch.gather(lora_win_log_prob_eval, dim=2, index=win_input_ids_shifted_eval.unsqueeze(-1)).squeeze(-1)
                    lora_lose_log_prob_gathered_eval = torch.gather(lora_lose_log_prob_eval, dim=2, index=lose_input_ids_shifted_eval.unsqueeze(-1)).squeeze(-1)
                    win_mask_eval = torch.roll(win_labels_eval != tokenizer.pad_token_id, shifts=-1, dims=1) # shift the mask to align with the log probs of the next tokens
                    lose_mask_eval = torch.roll(lose_labels_eval != tokenizer.pad_token_id, shifts=-1, dims=1)
                    lora_win_sentence_log_prob_eval = (lora_win_log_prob_gathered_eval * win_mask_eval)[:, :-1].sum(dim=1)
                    lora_lose_sentence_log_prob_eval = (lora_lose_log_prob_gathered_eval * lose_mask_eval)[:, :-1].sum(dim=1)
                    
                    with model.disable_adapter(): # make sure to disable adapter and set eval mode for reference model evaluation!
                        model.eval()
                        ref_win_outputs_eval = model(**win_model_inputs_eval, labels=None, use_cache=False)
                        ref_lose_outputs_eval = model(**lose_model_inputs_eval, labels=None, use_cache=False)
                        ref_win_log_prob_eval = torch.nn.functional.log_softmax(ref_win_outputs_eval.logits, dim=-1)
                        ref_lose_log_prob_eval = torch.nn.functional.log_softmax(ref_lose_outputs_eval.logits, dim=-1)
                        ref_win_log_prob_gathered_eval = torch.gather(ref_win_log_prob_eval, dim=2, index=win_input_ids_shifted_eval.unsqueeze(-1)).squeeze(-1)
                        ref_lose_log_prob_gathered_eval = torch.gather(ref_lose_log_prob_eval, dim=2, index=lose_input_ids_shifted_eval.unsqueeze(-1)).squeeze(-1)
                        ref_win_sentence_log_prob_eval = (ref_win_log_prob_gathered_eval * win_mask_eval)[:, :-1].sum(dim=1)
                        ref_lose_sentence_log_prob_eval = (ref_lose_log_prob_gathered_eval * lose_mask_eval)[:, :-1].sum(dim=1)
                    
                    loss = -torch.log(
                        torch.sigmoid(
                            BETA * (
                                lora_win_sentence_log_prob_eval - lora_lose_sentence_log_prob_eval
                                - ref_win_sentence_log_prob_eval + ref_lose_sentence_log_prob_eval
                            )
                        )
                    )
                    loss = torch.mean(loss)
                    eval_loss += loss.detach().item()
                    eval_batch_count += 1
                    
                    # compute KL divergence between current model and reference model
                    kl_div += torch.nn.functional.kl_div(
                        lora_win_log_prob_eval, ref_win_log_prob_eval, reduction='batchmean', log_target=True
                    ) + torch.nn.functional.kl_div(
                        lora_lose_log_prob_eval, ref_lose_log_prob_eval, reduction='batchmean', log_target=True
                    )
            kl_div /= eval_batch_count
            kl_div_history.append(kl_div.detach().item() / 2.0)
            eval_loss /= eval_batch_count
            eval_loss_history.append(eval_loss)
            eval_loss_steps.append(global_step)
            
            
            # update best eval loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model.save_pretrained(f"checkpoints/{model_name}_dpo_step_{global_step}")
                tokenizer.save_pretrained(f"checkpoints/{model_name}_dpo_step_{global_step}")
                print(f"New best model saved with eval loss {best_eval_loss:.4f}")
            
            # now, generate for several prompts in test set and evaluate the reward win rate against reference model
            win_count = 0
            total_count = 0
            # use the first few prompts
            pbar_eval = tqdm(test_dataloader, desc=f"Win Eval at Step {global_step}", leave=False)
            for batch_eval in pbar_eval:
                if total_count > REWARD_EVAL_PROMPT_COUNT:
                    break
                prompts = batch_eval["prompt"]
                # apply prompt processing
                processed_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
                # remember to use left padding for batched generation!
                inputs = tokenizer(processed_prompts, return_tensors="pt", padding_side='left', padding=True, truncation=True, max_length=MAX_LENGTH).to(model.device)
                with torch.no_grad():
                    model.eval()
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=3072,
                        do_sample=False, # deterministic decoding for reward evaluation
                    )
                    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
                    reward_inputs = reward_tokenizer(generated_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(next(reward_model.parameters()).device)
                    # manually set attention mask to ignore eof (id == 151643)
                    reward_inputs['attention_mask'][reward_inputs['input_ids'] == 151643] = 0
                    rewards = reward_model(reward_inputs)
                # compute reward for reference model outputs
                with torch.no_grad():
                    with model.disable_adapter():
                        model.eval()
                        ref_generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=3072,
                        )
                    ref_generated_texts = tokenizer.batch_decode(ref_generated_ids, skip_special_tokens=False)
                    ref_reward_inputs = reward_tokenizer(ref_generated_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(next(reward_model.parameters()).device)
                    # manually set attention mask to ignore eof (id == 151643)
                    ref_reward_inputs['attention_mask'][ref_reward_inputs['input_ids'] == 151643] = 0
                    ref_rewards = reward_model(ref_reward_inputs)
                # compute win rate
                win_count += torch.sum((rewards > ref_rewards).float()).item()
                total_count += rewards.size(0)
                if total_count == rewards.size(0):
                    # print example generations and rewards for the first prompt
                    print(f"Example Generation at Step {global_step}:")
                    print(f"Prompt: {prompts[0]}")
                    print(f"Reward: {rewards[0].item():.4f}, Reference Reward: {ref_rewards[0].item():.4f}")
                    print(f"Generated Text: {generated_texts[0]}")
                    print(f"Reference Generated Text: {ref_generated_texts[0]}")
            eval_win_rate = win_count / total_count if total_count > 0 else 0
            win_rate_history.append(eval_win_rate)
            
            tqdm.write(f"Step {global_step}: Train Loss={train_loss_history[-1]:.4f}, Eval Loss={eval_loss:.4f}, Win Rate={eval_win_rate:.4f}")
            
            plot_loss_curves()
        
    
        global_step += 1
    if accumulation_step % GRADIENT_ACCUMULATION_STEPS != 0:
        model.train()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    epoch_end_steps.append(len(train_loss_history))
    



# %%
# # for quick debug: shape check
# prompt = "Describe the property of sin and cos functions. List the properties one by one."
# text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").to(model.device)
# model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# shifted_input_ids = torch.roll(model_inputs["input_ids"], shifts=-1, dims=1)
# print(f"Shifted input IDs: {shifted_input_ids}") # shift position for next-token prediction

# # forward passes
# output_shape = torch.Size([1, 16, 151936])
# print("Output shape:", output_shape)
# outputs = torch.rand(output_shape).to(model.device)
# softmax_outputs = torch.nn.functional.softmax(outputs, dim=-1)
# print("Softmax output", softmax_outputs)
# log_prob = torch.log(softmax_outputs + 1e-8)
# gathered_log_prob = torch.gather(log_prob, dim=2, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)[:, :-1]

# sentence_log_prob = gathered_log_prob.sum(dim=1)
# print("Gathered log prob", gathered_log_prob)
# print("Sentence log prob", sentence_log_prob)


# %%
# import torch

# a = torch.tensor([[1, 2, 3]], dtype=torch.long)        # indices
# b = torch.tensor([[[0,1,2,3],
#                    [0,1,2,3],
#                    [0,1,2,3]]])                         # values

# # gather along last dim (dim=2); index must have same leading dims and an extra dim at gather axis
# res = torch.gather(b, dim=2, index=a.unsqueeze(-1)).squeeze(-1)
# print(res)  # tensor([[1,2,3]])


