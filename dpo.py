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
#test raw model
_prompt = "Describe the property of sin and cos functions. List the properties one by one."
_message = [{"role": "user", "content": _prompt}]
inputs = tokenizer.apply_chat_template(_message, add_generation_prompt=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
    )
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Raw Model Generation:\n{generated_text}\n")

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
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 1024
BETA = 0.1
LORA_RANK = 8
EXAMPLE_GEN_INTERVAL = 3000 # generate example outputs every EXAMPLE_GEN_INTERVAL optimization steps
EVAL_INTERVAL = 3000  # validate every 1000 steps

print(f"Hyperparameters:\n \
        Epochs: {EPOCHS}\n \
        Learning Rate: {LEARNING_RATE}\n \
        Batch Size: {BATCH_SIZE}\n \
        Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}\n \
        Max Sequence Length: {MAX_LENGTH}\n \
        Beta (DPO): {BETA}\n \
        LoRA Rank: {LORA_RANK}\n \
        Example Generation Frequency (steps): {EXAMPLE_GEN_INTERVAL}\n \
        Validation Frequency (steps): {EVAL_INTERVAL}\n \n")

# %%
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

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


# %%
# check out the injected lora layers!
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
print(f"Number of trainable parameters: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)}")
print(f"Number of trainable layers: {len(trainable)}")
print("Trainable layers:")
for item in trainable:
    print(item)

# %%
# get reference model by no_grad and temporarily disabling LoRA
prompt = "Describe the property of sin and cos functions. List the properties one by one."
text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt").to(model.device)
model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

policy_output = model(**model_inputs)

with torch.no_grad():
    with model.disable_adapter():
        ref_outputs = model(**model_inputs)

print("policy logits shape:", policy_output.logits.shape)
print("ref logits shape:", ref_outputs.logits.shape)
print(policy_output.logits)

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
kl_div_history = [] # compare the output distribution of the current model and the reference model at each validation step
kl_div_steps = [] 
best_eval_loss = float('inf')
global_step = 0

def plot_loss_curves():
    # plot training and eval loss curves
    train_loss_history_arr = np.array(train_loss_history)
    eval_loss_history_arr = np.array(eval_loss_history)
    eval_loss_steps_arr = np.array(eval_loss_steps)
    step_num_history_arr = np.arange(1, len(train_loss_history) + 1)
    kl_div_history_arr = np.array(kl_div_history)
    kl_div_steps_arr = np.array(kl_div_steps)
    
    # plot and save the training and validation loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step_num_history_arr, train_loss_history_arr, label='Training Loss', alpha=0.7)
    ax.plot(eval_loss_steps_arr, eval_loss_history_arr, label='Validation Loss', marker='o', markersize=4, alpha=0.7)
    
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

    # add right y-axis for KL divergence
    if len(kl_div_history_arr) > 0:
        ax2 = ax.twinx()
        ax2.plot(kl_div_steps_arr, kl_div_history_arr, label='KL Divergence', color='purple', marker='s', markersize=4, alpha=0.7)
        print(f"kl_div {kl_div_steps_arr, kl_div_history_arr}")
        ax2.set_ylabel('KL Divergence')
        ax2.legend(loc='upper right')
    
    # add a horizontal line showing minimum val loss
    ax.axhline(y=best_eval_loss, color='black', linestyle='--', linewidth=1, alpha=0.2, label=f'Min Eval Loss: {best_eval_loss:.4g}')

    ax.set_xlabel('Step Number')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curves for {model_name.split("/")[-1]} DPO Training')
    ax.legend()
    ax.grid(False)
    # # change y to log scale
    # ax.set_yscale('log')
    # ax.set_ylim(0.0001, 0.01)
    fig.tight_layout()
    fig.savefig(f"dpo_loss_curve.png")
    
for epoch in range(EPOCHS):
    # train
    pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{EPOCHS}")
    optimizer.zero_grad()
    accumulation_step = 0
    for batch in pbar:
        win_model_inputs = prepare_inputs(batch["chosen"], tokenizer, model.device)
        win_labels = mask_non_assistant_response(win_model_inputs, tokenizer)
        lose_model_inputs = prepare_inputs(batch["rejected"], tokenizer, model.device)
        lose_labels = mask_non_assistant_response(lose_model_inputs, tokenizer)
        win_input_ids_shifted = torch.roll(win_model_inputs["input_ids"], shifts=-1, dims=1)
        lose_input_ids_shifted = torch.roll(lose_model_inputs["input_ids"], shifts=-1, dims=1)
        
        # forward passes
        lora_win_outputs = model(**win_model_inputs, labels=None, use_cache=False) # disable KV cache
        lora_lose_outputs = model(**lose_model_inputs, labels=None, use_cache=False) # disable KV cache
        lora_win_softmax = torch.nn.functional.softmax(lora_win_outputs.logits, dim=-1)
        lora_lose_softmax = torch.nn.functional.softmax(lora_lose_outputs.logits, dim=-1)
        # compute sentence-level probability for responses by summing log probs (masked by labels)
        lora_win_log_prob = torch.log(lora_win_softmax + 1e-8)
        lora_lose_log_prob = torch.log(lora_lose_softmax + 1e-8)
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
                ref_win_outputs = model(**win_model_inputs, labels=None, use_cache=False)
                ref_lose_outputs = model(**lose_model_inputs, labels=None, use_cache=False)
                ref_win_softmax = torch.nn.functional.softmax(ref_win_outputs.logits, dim=-1)
                ref_lose_softmax = torch.nn.functional.softmax(ref_lose_outputs.logits, dim=-1)
                ref_win_log_prob = torch.log(ref_win_softmax + 1e-8)
                ref_lose_log_prob = torch.log(ref_lose_softmax + 1e-8)
                ref_win_log_prob_gathered = torch.gather(ref_win_log_prob, dim=2, index=win_input_ids_shifted.unsqueeze(-1)).squeeze(-1)
                ref_lose_log_prob_gathered = torch.gather(ref_lose_log_prob, dim=2, index=lose_input_ids_shifted.unsqueeze(-1)).squeeze(-1)
                # win_mask and lose_mask are the same for LoRA and reference since they are based on the input labels
                ref_win_sentence_log_prob = (ref_win_log_prob_gathered * win_mask)[:, :-1].sum(dim=1)
                ref_lose_sentence_log_prob = (ref_lose_log_prob_gathered * lose_mask)[:, :-1].sum(dim=1)
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
        global_step += 1
        
        # Step-based validation
        if global_step % EVAL_INTERVAL == 0:
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
                    lora_win_outputs_eval = model(**win_model_inputs_eval, labels=None, use_cache=False) # disable KV cache
                    lora_lose_outputs_eval = model(**lose_model_inputs_eval, labels=None, use_cache=False) # disable KV cache
                    ref_win_outputs_eval = model(**win_model_inputs_eval, labels=None, use_cache=False)
                    ref_lose_outputs_eval = model(**lose_model_inputs_eval, labels=None, use_cache=False)
                    lora_win_softmax_eval = torch.nn.functional.softmax(lora_win_outputs_eval.logits, dim=-1)
                    lora_lose_softmax_eval = torch.nn.functional.softmax(lora_lose_outputs_eval.logits, dim=-1)
                    ref_win_softmax_eval = torch.nn.functional.softmax(ref_win_outputs_eval.logits, dim=-1)
                    ref_lose_softmax_eval = torch.nn.functional.softmax(ref_lose_outputs_eval.logits, dim=-1)
                    lora_win_log_prob_eval = torch.log(lora_win_softmax_eval + 1e-8)
                    lora_lose_log_prob_eval = torch.log(lora_lose_softmax_eval + 1e-8)
                    ref_win_log_prob_eval = torch.log(ref_win_softmax_eval + 1e-8)
                    ref_lose_log_prob_eval = torch.log(ref_lose_softmax_eval + 1e-8)
                    lora_win_log_prob_gathered_eval = torch.gather(lora_win_log_prob_eval, dim=2, index=win_input_ids_shifted_eval.unsqueeze(-1)).squeeze(-1)
                    lora_lose_log_prob_gathered_eval = torch.gather(lora_lose_log_prob_eval, dim=2, index=lose_input_ids_shifted_eval.unsqueeze(-1)).squeeze(-1)
                    ref_win_log_prob_gathered_eval = torch.gather(ref_win_log_prob_eval, dim=2, index=win_input_ids_shifted_eval.unsqueeze(-1)).squeeze(-1)
                    ref_lose_log_prob_gathered_eval = torch.gather(ref_lose_log_prob_eval, dim=2, index=lose_input_ids_shifted_eval.unsqueeze(-1)).squeeze(-1)
                    win_mask_eval = torch.roll(win_labels_eval != tokenizer.pad_token_id, shifts=-1, dims=1) # shift the mask to align with the log probs of the next tokens
                    lose_mask_eval = torch.roll(lose_labels_eval != tokenizer.pad_token_id, shifts=-1, dims=1)
                    lora_win_sentence_log_prob_eval = (lora_win_log_prob_gathered_eval * win_mask_eval)[:, :-1].sum(dim=1)
                    lora_lose_sentence_log_prob_eval = (lora_lose_log_prob_gathered_eval * lose_mask_eval)[:, :-1].sum(dim=1)
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
            kl_div_steps.append(global_step)
            eval_loss /= eval_batch_count
            eval_loss_history.append(eval_loss)
            eval_loss_steps.append(global_step)
            tqdm.write(f"Step {global_step}: Train Loss={train_loss_history[-1]:.4f}, Eval Loss={eval_loss:.4f}")
            
            # update best eval loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                model.save_pretrained(f"checkpoints/{model_name}_dpo_step_{global_step}")
                tokenizer.save_pretrained(f"checkpoints/{model_name}_dpo_step_{global_step}")
                print(f"New best model saved with eval loss {best_eval_loss:.4f}")
            
            plot_loss_curves()
        
        if global_step % EXAMPLE_GEN_INTERVAL == 0:
            # example generation
            _prompt = "Describe the property of sin and cos functions. List the properties one by one."
            _message = [{"role": "user", "content": _prompt}]
            inputs = tokenizer.apply_chat_template(_message, add_generation_prompt=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True, 
                    temperature=0.7,
                    top_p=0.9,
                )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Example Generation at Epoch {epoch + 1} Step {accumulation_step + 1}:\n{generated_text}\n")

    if accumulation_step % GRADIENT_ACCUMULATION_STEPS != 0:
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


