
from pathlib import Path
import pandas as pd

# File locations
PROJECT_DIR = Path(__file__).resolve().parents[1]

SOURCE = PROJECT_DIR.parent / "raw_data" / "dataset.csv"

DESTINATION = PROJECT_DIR / "data" / "real_reviews_sample.csv"

if not SOURCE.is_file():
    raise FileNotFoundError(f"Cannot find source CSV: {SOURCE}")

# 1. Load real customer reviews
raw = pd.read_csv(
    SOURCE,
    usecols=["Review", "Rating"]
)

print(f"Original rows: {len(raw):,}")

# 2. Clean review text
clean = raw.copy()

clean["review"] = (
    clean["Review"]
    .astype("string")
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# 3. Convert ratings to numbers
clean["rating"] = pd.to_numeric(
    clean["Rating"],
    errors="coerce"
)

# 4. Remove invalid and very short reviews
clean = clean.loc[
    clean["review"].notna()
    & clean["rating"].between(1, 5)
    & (clean["review"].str.len() >= 25)
].copy()

# 5. Remove duplicate review texts
clean = clean.drop_duplicates(
    subset=["review"]
).reset_index(drop=True)

# 6. Randomly sample 2,000 reviews
sample_count = min(2000, len(clean))

sample = clean.sample(
    n=sample_count,
    random_state=42
).reset_index(drop=True)

# 7. Add unique review IDs
sample["review_id"] = [
    f"APPSTORE-{i:05d}"
    for i in range(1, sample_count + 1)
]

# The source does not identify the individual apps
sample["product"] = "Mixed App Store Reviews (app unknown)"

# 8. Export the cleaned sample
output = sample[
    ["review_id", "product", "rating", "review"]
]

output.to_csv(
    DESTINATION,
    index=False,
    encoding="utf-8"
)

print(f"Clean eligible rows: {len(clean):,}")
print(f"Sampled rows: {len(output):,}")
print(f"Saved: {DESTINATION}")

print("\nData preparation completed successfully!")
