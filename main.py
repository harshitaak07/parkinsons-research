import torch
from src.preprocessing.axivity_gait import preprocess_axivity
from src.preprocessing.opals_gait import preprocess_opals
from src.encoders.gait_encoder import GaitEncoder
#from src.encoders.non_motor_encoder import NonMotorEncoder
from src.encoders.time_embedding import TimeEmbedding
from src.fusion.intermediate_fusion import IntermediateFusion
from src.models.transform_classifier import TransformerClassifier
#from src.models.transformer_classifier import TransformerClassifier

df_axivity = preprocess_axivity("data/raw/Gait_Data___Arm_swing__Axivity__06Sep2025.csv", save=False)
df_opals   = preprocess_opals("data/raw/Gait_Data___Arm_swing__Opals__07Aug2025.csv", save=False)

df_axivity = df_axivity.head(10)
df_opals   = df_opals.head(10)

gait_features = df_axivity.select_dtypes(include='number').values
non_motor_features = df_opals.select_dtypes(include='number').values

delta_t = torch.rand(len(df_axivity), 1)

gait_encoder = GaitEncoder(input_dim=gait_features.shape[1])
#non_motor_encoder = NonMotorEncoder(input_dim=non_motor_features.shape[1])
time_encoder = TimeEmbedding(embedding_dim=16)

gait_emb = gait_encoder(torch.tensor(gait_features, dtype=torch.float32))
#non_motor_emb = non_motor_encoder(torch.tensor(non_motor_features, dtype=torch.float32))
time_emb = time_encoder(delta_t)

fusion = IntermediateFusion(mask_missing=False)
fused_emb = fusion(gait_emb, time_emb)

fused_emb_seq = fused_emb.unsqueeze(1)
transformer = TransformerClassifier(input_dim=fused_emb.shape[1])
output = transformer(fused_emb_seq)

print("Gait embedding shape:", gait_emb.shape)
#print("Non-motor embedding shape:", non_motor_emb.shape)
print("Time embedding shape:", time_emb.shape)
print("Fused embedding shape:", fused_emb.shape)
print("Transformer output shape (PD risk score):", output.shape)
