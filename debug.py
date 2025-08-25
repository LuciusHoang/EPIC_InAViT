import json, pandas as pd

pred_path = "results/inavit_predictions.jsonl"  # or your actual path
gt_csv = "EPIC_100_test_labels.csv"

# 1) Allowed action IDs from GT (only those present in your test subset)
df = pd.read_csv(gt_csv)
id_col = next(c for c in ["uid","narration_id","segment_id","id"] if c in df.columns)
act_col = next(c for c in ["action_id","action_cls","action"] if c in df.columns)
df = df[[id_col, act_col]].dropna()
df[id_col] = df[id_col].astype(str)
df[act_col] = df[act_col].astype(int)

allowed_ids = set(df[act_col].unique())
allowed_by_sid = dict(zip(df[id_col], df[act_col]))

# 2) Load predictions
preds = {}
with open(pred_path, "r") as f:
    for line in f:
        rec = json.loads(line)
        sid = str(rec.get("segment_id") or rec.get("uid") or rec.get("narration_id") or rec.get("id"))
        t5 = rec.get("action_top5_idx")
        preds[sid] = t5 if isinstance(t5, list) else None

# 3) Which samples would be dropped by an "allowed-ID" filter?
skipped = []
problem_samples = []  # keep why
for sid, t5 in preds.items():
    if t5 is None or len(t5) == 0:   # missing top-5 -> would skip
        skipped.append(sid)
        problem_samples.append((sid, "no_top5"))
        continue
    # remove anything not in allowed set
    filt = [int(x) for x in t5 if int(x) in allowed_ids]
    if len(filt) == 0:               # entire top-5 out-of-set -> would skip
        skipped.append(sid)
        problem_samples.append((sid, f"all_out_of_set: {t5}"))
        continue

print("Total preds:", len(preds))
print("Allowed unique action IDs in GT:", len(allowed_ids))
print("Would be evaluated (by this rule):", len(preds) - len(skipped))
print("Skipped count:", len(skipped))
print("Skipped sids + reason:")
for s, why in problem_samples:
    print(s, "=>", why)