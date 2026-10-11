"""Step 1.1: look at the text-to-SQL dataset before doing anything else."""
import json

from datasets import load_dataset

ds = load_dataset("b-mc2/sql-create-context", split="train")

print("rows:", len(ds))
print("columns:", ds.column_names)
for i in range(3):
    print(f"\n--- row {i} ---")
    print(json.dumps(ds[i], indent=2))
