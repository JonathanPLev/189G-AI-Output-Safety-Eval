from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")

device = "cuda"
print(dataset.keys())
print(dataset["train"].features)
print(dataset["train"].column_names)
differences = dataset["train"].filter(
    lambda x: x["is_response_0_safe"] != x["is_response_1_safe"]
)

count = 0
for difference in differences:
    print(
        f"prompt: {difference['prompt']} \nresponse 0: {difference['response_0']}\nresponse 1:{difference['response_1']}\nmalicious response: {1 if difference['is_response_0_safe'] == 1 else 0}\n"
    )
    count += 1
    if count >= 5:
        break
print(f"total difference: {len(differences)}")
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip(
    "\n"
)
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
