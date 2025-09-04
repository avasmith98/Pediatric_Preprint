import pandas as pd
import requests
import random
import os 
from time import sleep

### 1. Load and Clean Orange Book Data ###
file_path = "products.txt"
df = pd.read_csv(file_path, sep='~')

# Set column names
df.columns = [
    "Ingredient", "DF_Route", "Trade_Name", "Applicant", "Strength",
    "Appl_Type", "Appl_No", "Product_No", "TE_Code", "Approval_Date",
    "RLD", "RS", "Type", "Applicant_Full_Name"
]

# Drop discontinued and incomplete rows
df = df[~df["Type"].str.contains("DISC", case=False, na=False)]
df = df.dropna(subset=["Appl_No", "Appl_Type", "Trade_Name", "Ingredient"])

### 2. Load Random Seed and Sample Ingredients ###
# Check for seed file or create a new one
seed_file = "random_seed.txt"
if os.path.exists(seed_file):
    with open(seed_file) as f:
        seed = int(f.read().strip())
else:
    seed = random.randint(0, 10**6)
    with open(seed_file, "w") as f:
        f.write(str(seed))
    print(f"🆕 Created new seed: {seed} (saved to {seed_file})")

random.seed(seed)

# Sample 300 unique active ingredients
unique_ingredients = df["Ingredient"].drop_duplicates().sample(n=300, random_state=seed)
df_sampled = df[df["Ingredient"].isin(unique_ingredients)].copy()

# Save sampled entries
df_sampled.to_csv("sampled_ingredients_expanded.csv", index=False)
print("✔️ Saved sampled_ingredients_expanded.csv")

### 3. Prepare Queries ###
df_sampled["Appl_No"] = df_sampled["Appl_No"].astype(int).astype(str).str.zfill(6)
df_sampled["Appl_Type"] = df_sampled["Appl_Type"].map({"N": "NDA", "A": "ANDA"})
df_sampled = df_sampled.dropna(subset=["Appl_Type"])
df_sampled["application_number"] = (df_sampled["Appl_Type"] + df_sampled["Appl_No"]).str.upper()
df_sampled["brand_name"] = df_sampled["Trade_Name"].str.upper()

# Drop duplicate queries
query_df = df_sampled[["application_number", "Ingredient", "brand_name"]].drop_duplicates()
query_df.to_csv("fda_label_queries.csv", index=False)
print("✔️ Saved fda_label_queries.csv")

### 4. Query FDA API for Pediatric Use (One Per Ingredient) ###
all_records = []
seen_ingredients = set()

print("🔍 Searching for pediatric-use labels...")

for _, row in query_df.iterrows():
    ingredient = row["Ingredient"]
    if ingredient in seen_ingredients:
        continue
    if len(seen_ingredients) >= 100:
        break

    app_number = row["application_number"]
    brand = row["brand_name"]

    url = (
        f"https://api.fda.gov/drug/label.json"
        f"?search=openfda.application_number:\"{app_number}\""
        f"&limit=100"
    )

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        for result in data.get("results", []):
            fda_brands = result.get("openfda", {}).get("brand_name", [])
            if brand not in [b.upper() for b in fda_brands]:
                continue

            ped_text = result.get("pediatric_use", [""])[0]
            if ped_text.strip():
                all_records.append({
                    "application_number": app_number,
                    "ingredient": ingredient,
                    "brand_name": brand,
                    "full_ndc": result.get("openfda", {}).get("package_ndc", [""])[0],
                    "effective_time": result.get("effective_time"),
                    "version": result.get("version", ""),
                    "pediatric_use": ped_text
                })
                seen_ingredients.add(ingredient)
                break  # 🛑 Stop after first pediatric-use label is found

    except requests.HTTPError as e:
        print(f"⚠️ HTTP error for {brand} ({app_number}): {e}")
    except Exception as e:
        print(f"⚠️ Other error for {brand} ({app_number}): {e}")

    sleep(0.1)

print(f"✅ Collected pediatric-use info for {len(seen_ingredients)} ingredients")

### 5. Save Final Results ###
df_labels = pd.DataFrame(all_records)
df_labels["effective_time"] = pd.to_datetime(df_labels["effective_time"], errors="coerce")

# Save to CSV (each row = one unique ingredient)
df_labels.to_csv("latest_fda_labels_with_pediatric_use.csv", index=False)
print("✔️ Saved latest_fda_labels_with_pediatric_use.csv")
