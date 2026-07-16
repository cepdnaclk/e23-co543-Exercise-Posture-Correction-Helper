import pandas as pd

# Load the combined data
df = pd.read_csv("cleaned_exercise_baseline.csv")

# 1. Print all unique exercises and their frame counts
print("--- Actual Exercise Class Distribution ---")
# Replace 'pose_id' or 'exercise' with your actual label column name if it differs
# Let's check value counts for the target label column (usually named 'label' or 'class')
# If the column name is exactly 'label', use that:
if 'label' in df.columns:
    print(df['label'].value_counts())
elif 'pose_id' in df.columns:
    # If the dataset uses numeric IDs, let's see how many samples exist per ID
    print(df['pose'].value_counts())

# 2. List all angle columns available for Member 1 (Scoring Logic)
print("\n--- Available Pre-calculated Angles ---")
angle_cols = [col for col in df.columns if 'angle' in col or '_' in col and not col.startswith(('x', 'y', 'z', 'v'))]
print(angle_cols[:10]) # Prints the first 10 angle columns

# 1. Drop any rows with missing (NaN) values across the dataset
df_clean = df.dropna().copy()

print(f"\nOriginal rows: {df.shape[0]}")
print(f"Cleaned rows remaining: {df_clean.shape[0]}")

# 2. Save the final verified baseline dataset
df_clean.to_csv("cleaned_exercise_baseline.csv", index=False)
print("Saved final baseline dataset to 'cleaned_exercise_baseline.csv'")