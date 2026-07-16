import pandas as pd
import numpy as np

# Load your combined baseline data
df = pd.read_csv("cleaned_exercise_baseline.csv")

def normalize_pose_features(row):
    """
    Normalizes the x, y, z coordinates of a single frame 
    based on the user's torso size to eliminate camera distance bias.
    """
    try:
        # 1. Extract shoulder coordinates
        left_shoulder = np.array([row['x_left_shoulder'], row['y_left_shoulder'], row['z_left_shoulder']])
        right_shoulder = np.array([row['x_right_shoulder'], row['y_right_shoulder'], row['z_right_shoulder']])
        
        # 2. Extract hip coordinates
        left_hip = np.array([row['x_left_hip'], row['y_left_hip'], row['z_left_hip']])
        right_hip = np.array([row['x_right_hip'], row['y_right_hip'], row['z_right_hip']])
        
        # 3. Calculate midpoints
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        hip_mid = (left_hip + right_hip) / 2.0
        
        # 4. Calculate torso length (Euclidean distance) as the scale anchor
        torso_length = np.linalg.norm(shoulder_mid - hip_mid)
        
        # Prevent division by zero if tracking fails
        if torso_length == 0:
            torso_length = 1.0
            
        # 5. Scale all coordinate columns in this row by the torso length
        for col in row.index:
            if col.startswith(('x_', 'y_', 'z_')):
                row[col] = row[col] / torso_length
                
    except KeyError as e:
        print(f"Column missing error: {e}")
        
    return row

print("Starting spatial scale normalization...")
# Apply the normalization row by row across the dataset
df_normalized = df.apply(normalize_pose_features, axis=1)

# Save the scaled dataset
df_normalized.to_csv("normalized_exercise_features.csv", index=False)
print("Spatial normalization complete! Saved to 'normalized_exercise_features.csv'")