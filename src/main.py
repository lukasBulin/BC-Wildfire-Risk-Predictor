import subprocess
import sys

steps = [
    "src/api/fetch_data.py",
    "src/data/filter_bc_wildfires.py",
    "src/data/process_data.py",
    "src/model/train_model.py"
]

for step in steps:
    print(f"\nRunning {step}...\n")

    result = subprocess.run(
        [sys.executable, step]
    )

    if result.returncode != 0:
        print(f"Error while running {step}")
        break

print("\nPipeline complete!")