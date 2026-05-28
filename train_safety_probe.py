import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from datasets import load_dataset
from infer_danger_qwen_eval import model, tokenizer, hooks, layer_outputs

# WildGuardTrain for fitting, WildGuardTest for evaluation
train_ds = load_dataset("allenai/wildguardmix", "wildguardtrain")["train"].filter(
    lambda x: x["response"] is not None and x["response_harm_label"] is not None
)
test_ds = load_dataset("allenai/wildguardmix", "wildguardtest")["test"].filter(
    lambda x: x["response"] is not None and x["response_harm_label"] is not None
)

TRAIN_SAMPLE = 200
TEST_SAMPLE = 100

train_data = train_ds.select(range(min(TRAIN_SAMPLE, len(train_ds))))
test_data = test_ds.select(range(min(TEST_SAMPLE, len(test_ds))))


def collect_activations(data):
    X, y = [], []
    for item in data:
        layer_outputs["layer_12"] = []

        text = item["prompt"] + " " + item["response"]
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)

        if layer_outputs["layer_12"]:
            activation = layer_outputs["layer_12"][-1].numpy().flatten()
            X.append(activation)
            y.append(1 if item["response_harm_label"] == "harmful" else 0)

    return np.array(X), np.array(y)


print("Collecting train activations...")
X_train, y_train = collect_activations(train_data)

print("Collecting test activations...")
X_test, y_test = collect_activations(test_data)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print("\nSafety Probe — WildGuard test set results:")
print(classification_report(y_test, preds, target_names=["Unharmful", "Harmful"]))

for hook in hooks:
    hook.remove()
