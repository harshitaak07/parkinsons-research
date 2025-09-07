import torch
import numpy as np
from src.preprocessing.axivity_gait import preprocess_axivity
from src.preprocessing.opals_gait import preprocess_opals
from src.preprocessing.non_motor import preprocess_non_motor_with_labels
from src.encoders.gait_encoder import GaitEncoder
from src.encoders.non_motor_encoder import NonMotorEncoder
from src.encoders.time_embedding import TimeEmbedding
from src.fusion.intermediate_fusion import IntermediateFusion
from src.models.transform_classifier import TransformerClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
df_axivity = preprocess_axivity("data/raw/Gait_Data___Arm_swing__Axivity__06Sep2025.csv", save=False)
df_opals = preprocess_opals("data/raw/Gait_Data___Arm_swing__Opals__07Aug2025.csv", save=False)

# Load non-motor data (you'll need to provide the paths to your JSON folders)
# df_non_motor = preprocess_non_motor_with_labels(
#     questionnaire_folder="path/to/questionnaire/json/folder",
#     patients_folder="path/to/patients/json/folder",
#     save=False
# )

# Use more samples to see variation in predictions
df_axivity = df_axivity.head(50)
df_opals = df_opals.head(50)

gait_features = df_axivity.select_dtypes(include='number').values
non_motor_features = df_opals.select_dtypes(include='number').values

# For now, we'll use Opals data as proxy for non-motor features
# Later replace with actual non-motor data
delta_t = torch.rand(len(df_axivity), 1)

gait_encoder = GaitEncoder(input_dim=gait_features.shape[1])
non_motor_encoder = NonMotorEncoder(input_dim=non_motor_features.shape[1])
time_encoder = TimeEmbedding(embedding_dim=16)

fusion = IntermediateFusion(mask_missing=False)
transformer = TransformerClassifier(input_dim=48 + 32)  # 32 (gait) + 32 (non-motor) + 16 (time)

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
