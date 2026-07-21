import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================
print("Loading landmarks.csv and labels.csv...")
df_landmarks = pd.read_csv("landmarks.csv")
df_labels = pd.read_csv("labels.csv")

# Drop the 'pose_id' column so we only keep the 99 coordinate features
X = df_landmarks.drop(columns=['pose_id']).values

# Get the target labels from the 'pose' column
y_raw = df_labels['pose'].values

# Encode text labels (e.g., 'jumping_jacks_down') into integers (0, 1, 2...)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print(f"Dataset loaded! Features shape: {X.shape}")
print(f"Classes found ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

# Split into 80% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

class ExerciseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_loader = DataLoader(ExerciseDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(ExerciseDataset(X_val, y_val), batch_size=64, shuffle=False)

# ==========================================
# 2. DEFINE THE NEURAL NETWORK
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
# 3. TRAIN AND SAVE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}\n")

model = PoseRecognitionNet(input_dim=X.shape[1], num_classes=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 35
best_acc = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_x.size(0)
        correct += (out.argmax(dim=1) == batch_y).sum().item()
        total += batch_x.size(0)
        
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x)
            val_correct += (out.argmax(dim=1) == batch_y).sum().item()
            val_total += batch_x.size(0)
            
    val_acc = val_correct / val_total
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{epochs}] | Train Acc: {(correct/total)*100:.1f}% | Val Acc: {val_acc*100:.1f}%")
        
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_pose_model.pth")

print(f"\nTraining finished! Best accuracy achieved: {best_acc*100:.2f}%")

# Save preprocessing tools
joblib.dump(scaler, "pose_scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("SUCCESS: Saved 'best_pose_model.pth', 'pose_scaler.pkl', and 'label_encoder.pkl' to your folder!")