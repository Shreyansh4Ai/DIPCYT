import re
import yaml
import json
import csv
import os

task_log = r"C:\Users\goswa\.gemini\antigravity-ide\brain\4d857a71-0a01-4298-80a5-f1cac57a4859\.system_generated\tasks\task-511.log"
with open(task_log, "r") as f:
    text = f.read()

pattern = r"Trial (\d+) finished with value: ([\d\.\-]+) and parameters: (\{.*?\})\."
matches = re.findall(pattern, text)
print(f"Parsed {len(matches)} completed trials:")

trials_data = []
for m in matches:
    t_num = int(m[0])
    val = float(m[1])
    p = eval(m[2])
    row = {"trial_number": t_num, "value": val, **p}
    trials_data.append(row)
    print(f"  Trial {t_num:2d} | batch_size={p.get('batch_size')} | lr={p.get('lr'):.6f} | opt={p.get('optimizer')} | scheduler={p.get('scheduler')} | val={val}")

os.makedirs("optuna_results", exist_ok=True)

if trials_data:
    keys = list(trials_data[0].keys())
    with open("optuna_results/study_trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trials_data)

best_trial = max(trials_data, key=lambda x: (x["value"], x.get("optimizer") == "AdamW", x.get("batch_size") == 2 or x.get("batch_size") == 4))
best_dict = {k: v for k, v in best_trial.items() if k not in ("trial_number", "value")}

with open("optuna_results/best_params.yaml", "w") as f:
    yaml.dump(best_dict, f, default_flow_style=False)

print("\nSuccessfully generated optuna_results/best_params.yaml and study_trials.csv!")
print("Selected Best Configuration:")
print(yaml.dump(best_dict, default_flow_style=False))
