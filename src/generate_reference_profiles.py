import pandas as pd

# Load the normalized features (ensure you ran normalization_engine.py first)
try:
    df = pd.read_csv("normalized_exercise_features.csv")
except FileNotFoundError:
    df = pd.read_csv("cleaned_exercise_baseline.csv")
    print("Warning: Running profile analysis on unnormalized data. Run normalization first for final metrics.")

print("======================================================================")
# Focus analysis on the core joint pairs provided in the dataset
print("           EXERCISE FORM REFERENCE PROFILES (MEANS & EXTREMES)")
print("======================================================================\n")

# 1. Squats Analysis (Key Metrics: Knee Angles)
print("--- 1. SQUATS ---")
sq_down = df[df['pose'] == 'squats_down']
sq_up = df[df['pose'] == 'squats_up']

print("[Averages for Slides]")
print(f"  Squat Bottom (Peak Depth)  - Avg Right Knee: {sq_down['right_hip_right_knee_right_ankle'].mean():.2f}° | Avg Left Knee: {sq_down['left_hip_left_knee_left_ankle'].mean():.2f}°")
print(f"  Squat Standing (Extension) - Avg Right Knee: {sq_up['right_hip_right_knee_right_ankle'].mean():.2f}° | Avg Left Knee: {sq_up['left_hip_left_knee_left_ankle'].mean():.2f}°")
print("[Code Threshold Extremes]")
print(f"  Squat Bottom (Peak Depth)  - Min Right Knee: {sq_down['right_hip_right_knee_right_ankle'].min():.2f}° | Min Left Knee: {sq_down['left_hip_left_knee_left_ankle'].min():.2f}°")
print(f"  Squat Standing (Extension) - Max Right Knee: {sq_up['right_hip_right_knee_right_ankle'].max():.2f}° | Max Left Knee: {sq_up['left_hip_left_knee_left_ankle'].max():.2f}°\n")


# 2. Pushups Analysis (Key Metrics: Elbow Angles)
print("--- 2. PUSHUPS ---")
pu_down = df[df['pose'] == 'pushups_down']
pu_up = df[df['pose'] == 'pushups_up']

print("[Averages for Slides]")
print(f"  Pushup Bottom (Contraction)  - Avg Right Elbow: {pu_down['right_wrist_right_elbow_right_shoulder'].mean():.2f}° | Avg Left Elbow: {pu_down['left_wrist_left_elbow_left_shoulder'].mean():.2f}°")
print(f"  Pushup Top (Plank Position)  - Avg Right Elbow: {pu_up['right_wrist_right_elbow_right_shoulder'].mean():.2f}° | Avg Left Elbow: {pu_up['left_wrist_left_elbow_left_shoulder'].mean():.2f}°")
print("[Code Threshold Extremes]")
print(f"  Pushup Bottom (Contraction)  - Min Right Elbow: {pu_down['right_wrist_right_elbow_right_shoulder'].min():.2f}° | Min Left Elbow: {pu_down['left_wrist_left_elbow_left_shoulder'].min():.2f}°")
print(f"  Pushup Top (Plank Position)  - Max Right Elbow: {pu_up['right_wrist_right_elbow_right_shoulder'].max():.2f}° | Max Left Elbow: {pu_up['left_wrist_left_elbow_left_shoulder'].max():.2f}°\n")


# 3. Pullups Analysis (Key Metrics: Elbow Angles)
print("--- 3. PULLUPS ---")
pl_down = df[df['pose'] == 'pullups_down']
pl_up = df[df['pose'] == 'pullups_up']

print("[Averages for Slides]")
print(f"  Pullup Hanging (Extension)   - Avg Right Elbow: {pl_down['right_wrist_right_elbow_right_shoulder'].mean():.2f}° | Avg Left Elbow: {pl_down['left_wrist_left_elbow_left_shoulder'].mean():.2f}°")
print(f"  Pullup Top (Peak Contraction) - Avg Right Elbow: {pl_up['right_wrist_right_elbow_right_shoulder'].mean():.2f}° | Avg Left Elbow: {pl_up['left_wrist_left_elbow_left_shoulder'].mean():.2f}°")
print("[Code Threshold Extremes]")
print(f"  Pullup Hanging (Extension)   - Max Right Elbow: {pl_down['right_wrist_right_elbow_right_shoulder'].max():.2f}° | Max Left Elbow: {pl_down['left_wrist_left_elbow_left_shoulder'].max():.2f}°")
print(f"  Pullup Top (Peak Contraction) - Min Right Elbow: {pl_up['right_wrist_right_elbow_right_shoulder'].min():.2f}° | Min Left Elbow: {pl_up['left_wrist_left_elbow_left_shoulder'].min():.2f}°\n")


# 4. Situps Analysis (Key Metrics: Knee & Elbow Angles)
print("--- 4. SITUPS ---")
su_down = df[df['pose'] == 'situp_down']
su_up = df[df['pose'] == 'situp_up']

print("[Averages for Slides]")
print(f"  Situp Down (Mat Position) - Avg Left Knee: {su_down['left_hip_left_knee_left_ankle'].mean():.2f}° | Avg Left Elbow: {su_down['left_wrist_left_elbow_left_shoulder'].mean():.2f}°")
print(f"  Situp Up (Peak Squeeze)   - Avg Left Knee: {su_up['left_hip_left_knee_left_ankle'].mean():.2f}° | Avg Left Elbow: {su_up['left_wrist_left_elbow_left_shoulder'].mean():.2f}°")
print("[Code Threshold Extremes]")
print(f"  Situp Down (Mat Position) - Max Left Knee: {su_down['left_hip_left_knee_left_ankle'].max():.2f}°")
print(f"  Situp Up (Peak Squeeze)   - Min Left Knee: {su_up['left_hip_left_knee_left_ankle'].min():.2f}°\n")


# 5. Jumping Jacks Analysis (Key Metrics: Shoulder/Hip & Elbow Angles)
print("--- 5. JUMPING JACKS ---")
jj_down = df[df['pose'] == 'jumping_jacks_down']
jj_up = df[df['pose'] == 'jumping_jacks_up']

print("[Averages for Slides]")
print(f"  Jacks Down (Rest Position) - Avg Right Shoulder/Hip: {jj_down['right_elbow_right_shoulder_right_hip'].mean():.2f}° | Avg Left Elbow: {jj_down['left_wrist_left_elbow_left_shoulder'].mean():.2f}°")
print(f"  Jacks Up (Mid-Air Peak)    - Avg Right Shoulder/Hip: {jj_up['right_elbow_right_shoulder_right_hip'].mean():.2f}° | Avg Left Elbow: {jj_up['left_wrist_left_elbow_left_shoulder'].mean():.2f}°")
print("[Code Threshold Extremes]")
print(f"  Jacks Down (Rest Position) - Min Right Shoulder/Hip: {jj_down['right_elbow_right_shoulder_right_hip'].min():.2f}°")
print(f"  Jacks Up (Mid-Air Peak)    - Max Right Shoulder/Hip: {jj_up['right_elbow_right_shoulder_right_hip'].max():.2f}°\n")