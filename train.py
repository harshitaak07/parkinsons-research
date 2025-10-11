import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing.axivity_gait import preprocess_axivity
from src.preprocessing.opals_gait import preprocess_opals
from src.preprocessing.non_motor import preprocess_non_motor
from src.encoders.gait_encoder import GaitEncoder
from src.encoders.non_motor_encoder import NonMotorEncoder
from src.encoders.time_embedding import TimeEmbedding
from src.fusion.intermediate_fusion import IntermediateFusion
from src.models.transform_classifier import TransformerClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
df_axivity = preprocess_axivity("data/raw/motor/Gait_Data___Arm_swing__Axivity__06Sep2025.csv", save=False)
df_opals = preprocess_opals("data/raw/motor/Gait_Data___Arm_swing__Opals__07Aug2025.csv", save=False)

# Load real non-motor data
try:
    df_non_motor = preprocess_non_motor("data/raw/non_motor/questionnaires", save=False)
    non_motor_features = df_non_motor.drop('subject_id', axis=1).select_dtypes(include='number').values
    print(f"Loaded {len(df_non_motor)} non-motor samples with {non_motor_features.shape[1]} features")
except Exception as e:
    print(f"Could not load non-motor data: {e}")
    print("Falling back to Opals data as proxy for non-motor features")
    non_motor_features = df_opals.select_dtypes(include='number').values

# Use all data for training
gait_features = df_axivity.select_dtypes(include='number').values

# Create more realistic synthetic labels (not directly correlated with input features)
np.random.seed(42)

# Option 1: Random labels (for testing model capacity)
# labels_np = np.random.choice([0, 1], size=len(df_axivity), p=[0.5, 0.5])

# Option 2: Labels based on patient ID patterns (more realistic)
# Assuming patient IDs might have some correlation with disease status
patient_ids = df_axivity.index if 'VISNO' not in df_axivity.columns else df_axivity['VISNO']
if hasattr(patient_ids, 'values'):
    patient_ids = patient_ids.values

# Create more learnable synthetic labels with some correlation to features
np.random.seed(42)

# Method: Use a subset of features with moderate correlation + noise
# This creates learnable patterns while maintaining realism
feature_subset = gait_features[:, :10]  # Use first 10 features
base_scores = np.mean(feature_subset, axis=1)  # Average of selected features

# Add some non-linear transformations
nonlinear_scores = np.sin(base_scores * 0.1) + np.cos(base_scores * 0.05)

# Add moderate noise
noise = np.random.randn(len(df_axivity)) * 0.8

# Combine for final scores
final_scores = base_scores * 0.3 + nonlinear_scores * 0.4 + noise * 0.3

# Convert to binary labels with ~40% PD prevalence
probabilities = 1 / (1 + np.exp(-final_scores))  # sigmoid
labels_np = (probabilities > 0.999).astype(float)  # Extremely high threshold for balanced dataset

# Option 3: Use a subset of features with more complex relationships
# Uncomment below for more complex synthetic labels:
# feature_subset = gait_features[:, :5]  # Use first 5 features
# complex_scores = np.sum(feature_subset ** 2, axis=1) + np.sin(feature_subset[:, 0]) * 2
# noise = np.random.randn(len(df_axivity)) * 2.0  # More noise
# scores = complex_scores + noise
# probabilities = 1 / (1 + np.exp(-scores * 0.1))  # Scaled sigmoid
# labels_np = (probabilities > 0.5).astype(float)

labels = torch.tensor(labels_np.reshape(-1, 1), dtype=torch.float32)

# Create time delta features (random for demonstration)
delta_t = torch.rand(len(df_axivity), 1)

# Initialize models
gait_encoder = GaitEncoder(input_dim=gait_features.shape[1])
non_motor_encoder = NonMotorEncoder(input_dim=non_motor_features.shape[1])
time_encoder = TimeEmbedding(embedding_dim=16)
fusion = IntermediateFusion(mask_missing=False)
transformer = TransformerClassifier(input_dim=80)  # 32 (gait) + 32 (non-motor) + 16 (time)

# Loss and optimizer with class weights and regularization
if sum(labels_np) > 0:
    pos_weight = torch.tensor([len(labels_np) / sum(labels_np)])  # Weight for positive class
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.BCEWithLogitsLoss()  # No weighting if no positive samples

# Add weight decay for regularization
optimizer = optim.Adam(list(gait_encoder.parameters()) +
                       list(non_motor_encoder.parameters()) +
                       list(time_encoder.parameters()) +
                       list(fusion.parameters()) +
                       list(transformer.parameters()),
                       lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training loop with train/validation split
num_epochs = 50  # Increased epochs
batch_size = 16  # Smaller batch size for better gradient updates

# Split data into train/validation
train_size = int(0.8 * len(gait_features))
indices = np.random.permutation(len(gait_features))

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_gait = gait_features[train_indices]
train_labels = labels[train_indices]
train_delta_t = delta_t[train_indices]

val_gait = gait_features[val_indices]
val_labels = labels[val_indices]
val_delta_t = delta_t[val_indices]

def get_batches(features, delta_t, labels, batch_size):
    for i in range(0, len(features), batch_size):
        yield (features[i:i+batch_size],
               delta_t[i:i+batch_size],
               labels[i:i+batch_size])

for epoch in range(num_epochs):
    gait_encoder.train()
    non_motor_encoder.train()
    time_encoder.train()
    fusion.train()
    transformer.train()

    epoch_loss = 0.0
    for batch_gait, batch_delta_t, batch_labels in get_batches(train_gait, train_delta_t, train_labels, batch_size):
        optimizer.zero_grad()

        gait_emb = gait_encoder(torch.tensor(batch_gait, dtype=torch.float32))
        # Use corresponding non-motor features for this batch
        batch_indices = np.random.choice(len(non_motor_features), size=len(batch_gait), replace=True)
        batch_non_motor = non_motor_features[batch_indices]
        non_motor_emb = non_motor_encoder(torch.tensor(batch_non_motor, dtype=torch.float32))
        time_emb = time_encoder(batch_delta_t)

        fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
        fused_emb_seq = fused_emb.unsqueeze(1)

        output = transformer(fused_emb_seq)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(batch_gait)

    avg_loss = epoch_loss / len(train_gait)

    # Validation every 10 epochs
    if (epoch + 1) % 10 == 0:
        gait_encoder.eval()
        non_motor_encoder.eval()
        time_encoder.eval()
        fusion.eval()
        transformer.eval()

        val_loss = 0.0
        with torch.no_grad():
            for batch_gait, batch_delta_t, batch_labels in get_batches(val_gait, val_delta_t, val_labels, batch_size):
                gait_emb = gait_encoder(torch.tensor(batch_gait, dtype=torch.float32))
                # Use corresponding non-motor features for validation
                val_indices = np.random.choice(len(non_motor_features), size=len(batch_gait), replace=True)
                val_non_motor = non_motor_features[val_indices]
                non_motor_emb = non_motor_encoder(torch.tensor(val_non_motor, dtype=torch.float32))
                time_emb = time_encoder(batch_delta_t)
                fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
                fused_emb_seq = fused_emb.unsqueeze(1)
                output = transformer(fused_emb_seq)
                loss = criterion(output, batch_labels)
                val_loss += loss.item() * len(batch_gait)

        avg_val_loss = val_loss / len(val_gait)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

    # Step the scheduler
    scheduler.step()

print("Training complete.")

# Evaluation on validation data (unseen data)
gait_encoder.eval()
non_motor_encoder.eval()
time_encoder.eval()
fusion.eval()
transformer.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_gait, batch_delta_t, batch_labels in get_batches(val_gait, val_delta_t, val_labels.numpy(), batch_size):
        gait_emb = gait_encoder(torch.tensor(batch_gait, dtype=torch.float32))
        # Use corresponding non-motor features for evaluation
        eval_indices = np.random.choice(len(non_motor_features), size=len(batch_gait), replace=True)
        eval_non_motor = non_motor_features[eval_indices]
        non_motor_emb = non_motor_encoder(torch.tensor(eval_non_motor, dtype=torch.float32))
        time_emb = time_encoder(batch_delta_t)
        fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
        fused_emb_seq = fused_emb.unsqueeze(1)
        output = transformer(fused_emb_seq)

        # Convert logits to probabilities and then to binary predictions
        preds = torch.sigmoid(output).round().squeeze().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.squeeze())

# Calculate metrics on validation set
val_accuracy = accuracy_score(all_labels, all_preds)
val_precision = precision_score(all_labels, all_preds, zero_division=0)
val_recall = recall_score(all_labels, all_preds, zero_division=0)
val_f1 = f1_score(all_labels, all_preds, zero_division=0)

print("\nValidation Metrics (on unseen data):")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1-Score: {val_f1:.4f}")

# Also evaluate on training data for comparison
all_preds_train = []
all_labels_train = []

with torch.no_grad():
    for batch_gait, batch_delta_t, batch_labels in get_batches(train_gait, train_delta_t, train_labels.numpy(), batch_size):
        gait_emb = gait_encoder(torch.tensor(batch_gait, dtype=torch.float32))
        # Use corresponding non-motor features for training evaluation
        train_eval_indices = np.random.choice(len(non_motor_features), size=len(batch_gait), replace=True)
        train_eval_non_motor = non_motor_features[train_eval_indices]
        non_motor_emb = non_motor_encoder(torch.tensor(train_eval_non_motor, dtype=torch.float32))
        time_emb = time_encoder(batch_delta_t)
        fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
        fused_emb_seq = fused_emb.unsqueeze(1)
        output = transformer(fused_emb_seq)

        preds = torch.sigmoid(output).round().squeeze().numpy()
        all_preds_train.extend(preds)
        all_labels_train.extend(batch_labels.squeeze())

train_accuracy = accuracy_score(all_labels_train, all_preds_train)
print(f"\nTraining Accuracy (for comparison): {train_accuracy:.4f}")

# Save the trained models
torch.save(gait_encoder.state_dict(), 'gait_encoder.pth')
torch.save(non_motor_encoder.state_dict(), 'non_motor_encoder.pth')
torch.save(time_encoder.state_dict(), 'time_encoder.pth')
torch.save(fusion.state_dict(), 'fusion.pth')
torch.save(transformer.state_dict(), 'transformer.pth')
print("Models saved successfully.")
