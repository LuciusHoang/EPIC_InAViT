import pandas as pd

# Load CSVs for action segments
df_train = pd.read_csv("EK_data/EPIC_100_train.csv")
df_val = pd.read_csv("EK_data/EPIC_100_validation.csv")
df_full = pd.concat([df_train, df_val], ignore_index=True)

# Load class name mappings
verb_df = pd.read_csv("EK_data/EPIC_100_verb_classes.csv")  # columns: id, key
noun_df = pd.read_csv("EK_data/EPIC_100_noun_classes.csv")  # columns: id, key

# Extract unique (verb_class, noun_class) and assign action_id
unique_actions = df_full[['verb_class', 'noun_class']].drop_duplicates().reset_index(drop=True)
unique_actions['action_id'] = unique_actions.index

# Merge human-readable labels
unique_actions = unique_actions.merge(verb_df.rename(columns={'id': 'verb_class', 'key': 'verb'}), on='verb_class')
unique_actions = unique_actions.merge(noun_df.rename(columns={'id': 'noun_class', 'key': 'noun'}), on='noun_class')

# Build a dictionary: action_id → (verb, noun)
action_id_to_label = {
    row.action_id: (row.verb, row.noun)
    for _, row in unique_actions.iterrows()
}

# ✅ Preview
print(f"Total actions: {len(action_id_to_label)}")
print("🔍 Sample entries:")
for aid in list(action_id_to_label.keys())[:5]:
    print(f"  {aid}: {action_id_to_label[aid]}")

# Reorder and select columns for clarity
columns = ['action_id', 'verb_class', 'verb', 'noun_class', 'noun']
action_df = unique_actions[columns].sort_values(by='action_id')

# Save to CSV
action_df.to_csv("epic_ek100_action_labels.csv", index=False)
print("✅ Saved to epic_ek100_action_labels.csv")
