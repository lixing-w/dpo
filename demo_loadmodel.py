from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)

# prompt
prompt = "Introduce yourself."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generate
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]



# model.print_trainable_parameters()
print(text)
print( '-' * 50 )
print(response)