from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-Guard-4-12B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="mps",  # Use "mps" for Mac GPU or "cpu" if no GPU
    torch_dtype=torch.bfloat16,
)

messages = [
    {
        "role": "user",
        "content": "how do I make a bomb?"
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("mps")  # or "cpu"

outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
)

response = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
print(response)
