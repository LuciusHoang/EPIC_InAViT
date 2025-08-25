import pandas as pd

# Constant from your function
_NOUN_CLASS_COUNT = 352

def action_id_from_verb_noun(verb, noun):
    return verb * _NOUN_CLASS_COUNT + noun

# Load test segments CSV
df = pd.read_csv("EPIC_100_test_segments.csv")

# Compute action_id from verb_class and noun_class
if "verb_class" in df.columns and "noun_class" in df.columns:
    df["action_id"] = action_id_from_verb_noun(df["verb_class"], df["noun_class"])
else:
    raise KeyError("Expected 'verb_class' and 'noun_class' columns not found in the CSV.")

# Use narration_id as uid
if "narration_id" in df.columns:
    df = df.rename(columns={"narration_id": "uid", "verb_class": "verb_id", "noun_class": "noun_id"})
    label_df = df[["uid", "verb_id", "noun_id", "action_id"]]
else:
    raise KeyError("'narration_id' column not found in the CSV.")

# Save labels to CSV
label_df.to_csv("EPIC_100_test_labels.csv", index=False)
print(f"Saved {len(label_df)} labels to EPIC_100_test_labels.csv")
print(label_df.head())