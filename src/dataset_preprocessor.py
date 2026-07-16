import pandas as pd
import os

# Define the directory where your files are stored
data_dir = "archive"

# 1. Load the core files
print("Loading dataset components...")
df_labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))
df_landmarks = pd.read_csv(os.path.join(data_dir, "landmarks.csv"))
df_angles = pd.read_csv(os.path.join(data_dir, "angles.csv"))

# Verify shapes match before merging
print(f"Labels rows: {df_labels.shape[0]}")
print(f"Landmarks rows: {df_landmarks.shape[0]}")
print(f"Angles rows: {df_angles.shape[0]}")

# 2. Combine them horizontally (column-wise) assuming rows line up perfectly
# Using axis=1 joins the columns of dataframes sharing the same row index
df_combined = pd.concat([df_labels, df_landmarks, df_angles], axis=1)

print(f"\nCombined Dataset Shape: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns")

# 3. Inspect the first few rows to confirm successful merge
print("\nFirst 3 rows of the merged baseline dataset:")
print(df_combined.head(3))

# Save this out as a temporary working master file for the team
df_combined.to_csv("combined_raw_dataset.csv", index=False)
print("\nSaved unified data to 'combined_raw_dataset.csv'")