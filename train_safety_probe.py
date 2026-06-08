import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# We import the setup directly from your team's file without changing it
from infer_danger_qwen_eval import model, tokenizer, get_activation, hooks, dataset

print("Starting Safety Probe Training Extension...")

X_activations = []
y_labels = []

# Filter for split data where choices differ (safe vs unsafe)
evaluation_data = dataset["train"].filter(lambda x: x["is_response_0_safe"] != x["is_response_1_safe"]).select(range(50)) # Sample 50 to prototype

# Clear out default test run tracker
from infer_danger_qwen_eval import layer_outputs
layer_outputs["layer_12"] = []

for idx, item in enumerate(evaluation_data):
    # Check layer 12 hidden states for response_0
    prompt = item["prompt"] + " " + item["response_0"]
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)
        
    if layer_outputs["layer_12"]:
        # Grab the last token's activation vector
        last_token_activation = layer_outputs["layer_12"][-1].numpy().flatten()
        X_activations.append(last_token_activation)
        y_labels.append(item["is_response_0_safe"])
        
    # Flush for next iteration
    layer_outputs["layer_12"] = []

# Train a linear safety probe on top of layers
X = np.array(X_activations)
y = np.array(y_labels)

if len(X) > 0:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    preds = clf.predict(X)
    
    print("\n Safety Probe Performance Report on Hidden States:")
    print(classification_report(y, preds, target_names=["Unsafe", "Safe"]))
else:
    print("Failed to gather activations.")

# Clean up hooks cleanly
for hook in hooks:
    hook.remove()