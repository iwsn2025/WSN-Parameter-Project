import pandas as pd
import os
from exp_single import exp as exp_single
from exp_multi import exp as exp_multi
import logging

logging.basicConfig(level=logging.DEBUG)

def run_experiments():
    shots = list(range(1, 7))
    results = {"Shots": shots, "Single Source Accuracy": [], "Multi Source Accuracy": []}

    for i in shots:
        print(f"Running Single Source Experiment with {i} shot(s)...")
        _, test_acc = exp_single(shot=i, shuffle=False, epoch=400)
        results["Single Source Accuracy"].append(test_acc)

    for i in shots:
        print(f"Running Multi Source Experiment with {i} shot(s)...")
        _, test_acc = exp_multi(shot=i, shuffle=False, epoch=600)
        results["Multi Source Accuracy"].append(test_acc)

    return pd.DataFrame(results)

# Generate data and save to CSV
df = run_experiments()
csv_path = os.path.join(os.path.dirname(__file__), "experiment_results.csv")
df.to_csv(csv_path, index=False)
print(f"Experiment results saved to {csv_path}")