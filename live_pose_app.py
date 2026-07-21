import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. RECREATE THE MODEL ARCHITECTURE
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.gelu(out + residual)

class PoseRecognitionNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)

# ==========================================
# 2. LOAD SAVED WEIGHTS & SCALERS
# ==========================================
print("Loading model, scaler, and label encoder...")
scaler = joblib.load("pose_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

input_dim = scaler.n_features_in_  # Will be exactly 99
num_classes = len(label_encoder.classes_)  # Will be exactly 10

# Automatically detect compute device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Running real-time inference on device: {device}")

model = PoseRecognitionNet(input_dim=input_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("best_pose_model.pth", map_location=device))
model.eval()  # Put model into evaluation mode

# ==========================================
# 3. INITIALIZE WEBCAM & MEDIAPIPE
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open default camera

print("Webcam successfully started! Step back so your full body is visible.")
print("Press 'q' on your keyboard while clicked on the video window to quit.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Could not grab webcam frame.")
            break
            
        # Flip horizontally for a natural mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Detect body joints
        results = pose.process(image_rgb)
        
        # Convert back to BGR for drawing on the screen
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # ==========================================
        # 4. EXTRACT 99 FEATURES & PREDICT
        # ==========================================
        if results.pose_landmarks:
            # Draw the green/red skeleton overlay onto your joints
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
            
            # Extract only x, y, z coordinates for the 33 landmarks (33 * 3 = 99 features)
            raw_features = []
            for landmark in results.pose_landmarks.landmark:
                raw_features.extend([landmark.x, landmark.y, landmark.z])
                
            # Convert to numpy array, scale it, and turn it into a PyTorch Tensor
            feature_array = np.array(raw_features).reshape(1, -1)
            scaled_features = scaler.transform(feature_array)
            tensor_features = torch.tensor(scaled_features, dtype=torch.float32).to(device)
            
            # Predict the pose
            with torch.no_grad():
                logits = model(tensor_features)
                probabilities = F.softmax(logits, dim=1)
                max_prob, predicted_class_idx = torch.max(probabilities, dim=1)
            
            # Convert predicted integer back to text label
            exercise_name = label_encoder.inverse_transform([predicted_class_idx.cpu().item()])[0]
            confidence = max_prob.item() * 100
            
            # Format label for display (replace underscores with spaces and capitalize)
            display_text = f"POSE: {exercise_name.replace('_', ' ').upper()} ({confidence:.1f}%)"
            
            # Draw banner at top of video window
            cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
            cv2.putText(
                frame, 
                display_text, 
                (15, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 0) if confidence > 75 else (0, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
        # Display the video window
        cv2.imshow('Real-Time PyTorch Exercise Recognition', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
