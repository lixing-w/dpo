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
# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "Qwen/Qwen2.5-0.5B"
# or use a larger model
# model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
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
# set up torch device
if torch.cuda.is_available():
    device = torch.device( "cuda" )
elif torch.backends.mps.is_available():
    device = torch.device( "mps" )
else:
    device = torch.device( "cpu" )

# set up hyperparameters
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 1024

print(f"Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, Gradient Acc Steps: {GRADIENT_ACCUMULATION_STEPS}, Max Seq Len: {MAX_LENGTH}")

# %%
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# %%
# prepare the model for lora
# lora_config = LoraConfig(
#     r=8, # the rank
#     lora_alpha=16, # the scaling factor
#     lora_dropout=0.05,
#     target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# sft_model = get_peft_model(model, lora_config)
# sft_model.gradient_checkpointing_enable()
# sft_model.config.use_cache = False
# sft_model.print_trainable_parameters()


# %%
# model.save_pretrained(f"checkpoints/{model_name}_sft_epoch_0")
# model = AutoModelForCausalLM.from_pretrained(f"checkpoints/{model_name}_sft_epoch_0", device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa")

# %%
# setup full-finetuning training loop for SFT
from tqdm import tqdm
import torch.optim as optim

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

# set up optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS*len(train_dataloader), eta_min=1e-6)

train_loss_history = [] # for each step (batch)
eval_loss_history = [] # for each epoch (average over batches)
epoch_end_steps = [] # to keep track of step number at the end of each epoch for plotting
best_eval_loss = float('inf')
for epoch in range(EPOCHS):
    # train
    pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{EPOCHS}")
    optimizer.zero_grad()
    accumulation_step = 0
    for batch in pbar:
        conversations = unbatch_chat_messages(batch["messages"])
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in conversations
        ]
        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(model.device)
        labels = mask_non_assistant_response(model_inputs, tokenizer)
        # forward pass
        outputs = model(**model_inputs, labels=labels) # remember to use sft_model here, not the original model
        loss = outputs.loss
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        accumulation_step += 1
        if accumulation_step % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss_history.append(loss.detach().item())
        
        if accumulation_step % 1910 == 0:
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
    
    # eval
    pbar = tqdm(test_dataloader, desc=f"Eval Epoch {epoch + 1}/{EPOCHS}")
    eval_avg_loss = 0
    eval_batch_count = 0
    for batch in pbar:
        conversations = unbatch_chat_messages(batch["messages"])
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in conversations
        ]
        model_inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(model.device)
        labels = mask_non_assistant_response(model_inputs, tokenizer)
        with torch.no_grad():
            outputs = model(**model_inputs, labels=labels)
            loss = outputs.loss
            eval_avg_loss += loss.detach().item()
            eval_batch_count += 1
    eval_avg_loss /= eval_batch_count
    eval_loss_history.append(eval_avg_loss)
    tqdm.write(f"Epoch {epoch + 1} Eval Loss: {eval_avg_loss:.4f}")

    # update best eval loss
    if eval_avg_loss < best_eval_loss:
        best_eval_loss = eval_avg_loss
        # save state dict of the model
        model.save_pretrained(f"checkpoints/{model_name}_sft_epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"checkpoints/{model_name}_sft_epoch_{epoch + 1}")
        print(f"New best model saved with eval loss {best_eval_loss:.4f}")
    

# plot training and eval loss curves
# repeat eval_loss_history to match other histories for plotting
train_loss_history = np.array(train_loss_history)
eval_loss_history = np.array(eval_loss_history)
eval_loss_history = np.repeat(eval_loss_history, int(len(train_loss_history) / len(eval_loss_history)))
step_num_history = np.arange(1, len(train_loss_history) + 1)

# plot and save the training and validation loss curves
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(step_num_history, train_loss_history, label='Training Loss', alpha=0.7)
ax.plot(step_num_history, eval_loss_history, label='Validation Loss', alpha=0.7)

# add vertical lines at epoch boundaries
for step_idx in epoch_end_steps:
    ax.axvline(step_idx, color='gray', ls='--', lw=0.6, alpha=0.35)

# add epoch ticks on the top axis
if len(epoch_end_steps) > 0:
    epoch_end_steps = np.array(epoch_end_steps)
    epoch_ids = np.arange(1, len(epoch_end_steps) + 1)
    max_labels = 12
    stride = max(1, int(np.ceil(len(epoch_end_steps) / max_labels)))
    top_ticks = epoch_end_steps[::stride]
    top_labels = epoch_ids[::stride]
    top_ax = ax.secondary_xaxis('top')
    top_ax.set_xticks(top_ticks)
    top_ax.set_xticklabels(top_labels)
    top_ax.set_xlabel('Epoch')

# add a horizontal line showing minimum val loss
ax.axhline(y=best_eval_loss, color='black', linestyle='--', linewidth=1, alpha=0.2, label=f'Min Eval Loss: {best_eval_loss:.4g}')

ax.set_xlabel('Step Number')
ax.set_ylabel('Loss')
ax.set_title(f'Loss Curves for {model_name.split( "/" )[ -1 ]}')
ax.legend()
ax.grid(False)
# change y to log scale
ax.set_yscale('log')
# ax.set_ylim(0.0001, 0.01)
fig.tight_layout()
fig.savefig(f"sft_loss_curve.png")
plt.show()
            

# %%
# # check out the injected lora layers!
# trainable = [(n, p.shape) for n, p in policy_model.named_parameters() if p.requires_grad]
# print(f"Number of trainable parameters: {sum(p.numel() for n, p in policy_model.named_parameters() if p.requires_grad)}")
# print(f"Number of trainable layers: {len(trainable)}")
# print("Trainable layers:")
# for item in trainable:
#     print(item)

# %%
# # get reference model by no_grad and temporarily disabling LoRA
# policy_output = policy_model(**model_inputs)

# with torch.no_grad():
#     with policy_model.disable_adapter():
#         ref_outputs = policy_model(**model_inputs)

# print("policy logits shape:", policy_output.logits.shape)
# print("ref logits shape:", ref_outputs.logits.shape)


