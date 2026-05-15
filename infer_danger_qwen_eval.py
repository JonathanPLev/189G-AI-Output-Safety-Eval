from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "Qwen/Qwen3-0.6B"
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")

device = "cuda"

# # uncomment for dataset information
# print(dataset.keys())
# print(dataset["train"].features)
# print(dataset["train"].column_names)
# differences = dataset["train"].filter(
#     lambda x: x["is_response_0_safe"] != x["is_response_1_safe"]
# )


# # examine prompt responses in dataset
# count = 0
# for difference in differences:
#     print(
#         f"prompt: {difference['prompt']} \nresponse 0: {difference['response_0']}\nresponse 1:{difference['response_1']}\nmalicious response: {1 if difference['is_response_0_safe'] == 1 else 0}\n"
#     )
#     count += 1
#     if count >= 5:
#         break
# print(f"total difference: {len(differences)}")


MIN_LAYER_TO_COLLECT = 12
MAX_LAYER_TO_COLLECT = 13  # not inclusive
layer_outputs = {
    f"layer_{i}": [] for i in range(MIN_LAYER_TO_COLLECT, MAX_LAYER_TO_COLLECT)
}


# state saver hook
def get_activation(layer_name):
    def hook(model, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # during prefill, hidden states length is based on input,
        # we don't want to save b/c the tokens are based on prompt, not on generation
        # during generation the hidden state uses kv cache so seq_len will always be 1.
        if hidden_states.shape[1] == 1:  # [Batch, seq_len, hidden_dim]
            # only save the hidden state for the last token, since we are saving the hidden state at every token decoding anyway
            layer_outputs[layer_name].append(
                hidden_states[:, 0, :].detach().cpu()  # token sequence
            )  # squeeze, [Batch, hidden_dim]

    return hook


# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype="auto", device_map="auto"
# )

save_dir = "Qwen-0.6b"
# # uncomment to save model locally
# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)


# if saved locally
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(
    save_dir, torch_dtype="auto", device_map="auto", local_files_only=True
)


target_layers = range(MIN_LAYER_TO_COLLECT, MAX_LAYER_TO_COLLECT)
hooks = []

for layer_idx in target_layers:
    layer = model.model.layers[layer_idx]
    hook_handle = layer.register_forward_hook(get_activation(f"layer_{layer_idx}"))
    hooks.append(hook_handle)


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

print(f"total number of tokens: {len(output_ids)}")


print(f"Input token count: {model_inputs['input_ids'].shape[1]}\n")
for layer_name, tensor in layer_outputs.items():
    actual_tensor = tensor[0]
    print(f"type: {actual_tensor}")
    print(
        f"{layer_name} output shape: {list(actual_tensor.shape)}"
    )  # [batch_size, hidden_dimension]
    print(f"number of tokens for which layers are saved: {len(tensor)}")

for hook in hooks:
    hook.remove()
