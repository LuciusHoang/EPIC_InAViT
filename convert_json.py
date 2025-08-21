import json

input_file = "results/inavit_predictions.jsonl"
output_file = "results/inavit_predictions.json"

with open(input_file, "r") as f:
    data = [json.loads(line) for line in f]

# Safely rename clip_uid to uid if present
for item in data:
    if "clip_uid" in item:
        item["uid"] = item.pop("clip_uid")

with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"Converted {len(data)} predictions and saved to {output_file}")