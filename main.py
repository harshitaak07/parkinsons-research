import torch
import numpy as np
from motor.axivity_gait import preprocess_axivity
from motor.opals_gait import preprocess_opals
from nonmotor.non_motor import preprocess_non_motor
from motor.gait_encoder import GaitEncoder
from nonmotor.non_motor_encoder import NonMotorEncoder
from models.time_embedding import TimeEmbedding
from models.intermediate_fusion import IntermediateFusion
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

# Use more samples to see variation in predictions
df_axivity = df_axivity.head(50)
df_opals = df_opals.head(50)

gait_features = df_axivity.select_dtypes(include='number').values

# Use real non-motor data if available, otherwise use proxy
if 'df_non_motor' in locals() and len(df_non_motor) > 0:
    # Use real non-motor data
    non_motor_subset = df_non_motor.head(50)  # Match the sample size
    non_motor_features = non_motor_subset.drop('subject_id', axis=1).select_dtypes(include='number').values
    print("Using real non-motor data for inference")
else:
    # Fall back to proxy
    non_motor_features = df_opals.select_dtypes(include='number').values
    print("Using proxy non-motor data for inference")
delta_t = torch.rand(len(df_axivity), 1)

gait_encoder = GaitEncoder(input_dim=gait_features.shape[1])
non_motor_encoder = NonMotorEncoder(input_dim=non_motor_features.shape[1])
time_encoder = TimeEmbedding(embedding_dim=16)

fusion = IntermediateFusion(mask_missing=False)
transformer = TransformerClassifier(input_dim=80)  # 32 (gait) + 32 (non-motor) + 16 (time)

# Load trained model weights
try:
    gait_encoder.load_state_dict(torch.load('gait_encoder.pth'))
    non_motor_encoder.load_state_dict(torch.load('non_motor_encoder.pth'))
    time_encoder.load_state_dict(torch.load('time_encoder.pth'))
    fusion.load_state_dict(torch.load('fusion.pth'))
    transformer.load_state_dict(torch.load('transformer.pth'))
    print("Loaded trained model weights.")
except FileNotFoundError:
    print("No trained model weights found. Using untrained model.")

gait_encoder.eval()
non_motor_encoder.eval()
time_encoder.eval()
fusion.eval()
transformer.eval()

with torch.no_grad():
    gait_emb = gait_encoder(torch.tensor(gait_features, dtype=torch.float32))
    non_motor_emb = non_motor_encoder(torch.tensor(non_motor_features, dtype=torch.float32))
    time_emb = time_encoder(delta_t)
    fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
    fused_emb_seq = fused_emb.unsqueeze(1)
    output = transformer(fused_emb_seq)

print("Gait embedding shape:", gait_emb.shape)
print("Non-motor embedding shape:", non_motor_emb.shape)
print("Time embedding shape:", time_emb.shape)
print("Fused embedding shape:", fused_emb.shape)
print("Transformer output shape (PD risk score):", output.shape)
print("PD risk scores:", output.squeeze().tolist())
