import pandas as pd

df = pd.read_csv("cleaned_exercise_baseline.csv")

# Print all column names to see where the text labels are hidden
print("Columns in the dataset:")
print(list(df.columns))

# Look for columns named 'label', 'exercise', 'class', or 'pose' (excluding pose_id)
# Once you spot it, check its value distribution. For example, if it's named 'label':
if 'label' in df.columns:
    print("\nActual Exercise Breakdown:")
    print(df['label'].value_counts())