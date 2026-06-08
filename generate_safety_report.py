import json
import time
from datetime import datetime

def log_benchmark_run(model_name, input_tokens, generated_tokens, execution_time):
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "execution_time_sec": round(execution_time, 4),
        "tokens_per_second": round(generated_tokens / execution_time, 2) if execution_time > 0 else 0
    }
    
    # Save metrics to a local database file (ignored by your gitignore setup)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/benchmark_results.jsonl", "a") as f:
        f.write(json.dumps(report_data) + "\n")
        
    print(f"Metric logged for {model_name}.")

if __name__ == "__main__":
    print("Run alongside/after generation loop to map safety latencies.")