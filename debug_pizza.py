import pandas as pd
import json

# Load metadata
with open('data/sandiego_meta.json', 'r') as f:
    metadata = json.load(f)

df = pd.DataFrame.from_dict(metadata, orient='index')

# Find pizza places
df['cat_str'] = df['category'].astype(str).str.lower()
pizza_mask = df['cat_str'].str.contains('pizza', na=False, case=False)
pizza_df = df[pizza_mask]

print(f"Total pizza places: {len(pizza_df)}")

# Check addresses for ones that might be near La Jolla
keywords = ['la jolla', 'pacific beach', 'utc', 'university city', 'clairemont', 'del mar']
for kw in keywords:
    matches = pizza_df[pizza_df['address'].astype(str).str.lower().str.contains(kw, na=False)]
    if not matches.empty:
        print(f"\n=== {kw.upper()} ({len(matches)}) ===")
        for idx, row in matches.head(5).iterrows():
            print(f"  • {row['name']} - {row['address']}")
