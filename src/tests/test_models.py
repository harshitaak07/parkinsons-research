import torch
import numpy as np
<<<<<<< HEAD
from src.motor.gait_encoder import GaitEncoder
from src.nonmotor.non_motor_encoder import NonMotorEncoder
from src.models.time_embedding import TimeEmbedding
from src.models.intermediate_fusion import IntermediateFusion
=======
from motor.gait_encoder import GaitEncoder
from nonmotor.non_motor_encoder import NonMotorEncoder
from models.time_embedding import TimeEmbedding
from models.intermediate_fusion import IntermediateFusion
>>>>>>> motor-modalities
from src.models.transform_classifier import TransformerClassifier

print("Testing Model Components...")

batch_size = 10
gait_dim = 103
non_motor_dim = 30
time_dim = 1
gait_input = torch.randn(batch_size, gait_dim)
non_motor_input = torch.randn(batch_size, non_motor_dim)
time_input = torch.randn(batch_size, time_dim)

print("Testing Gait Encoder...")
try:
    gait_encoder = GaitEncoder(input_dim=gait_dim)
    gait_emb = gait_encoder(gait_input)
    print(f"Gait Encoder: input {gait_input.shape} → output {gait_emb.shape}")
except Exception as e:
    print(f"❌ Gait Encoder failed: {e}")

print("Testing Non-Motor Encoder...")
try:
    non_motor_encoder = NonMotorEncoder(input_dim=non_motor_dim)
    non_motor_emb = non_motor_encoder(non_motor_input)
    print(f"Non-Motor Encoder: input {non_motor_input.shape} → output {non_motor_emb.shape}")
except Exception as e:
    print(f"Non-Motor Encoder failed: {e}")

print("Testing Time Embedding...")
try:
    time_encoder = TimeEmbedding(embedding_dim=16)
    time_emb = time_encoder(time_input)
    print(f"Time Embedding: input {time_input.shape} → output {time_emb.shape}")
except Exception as e:
    print(f"Time Embedding failed: {e}")

print("Testing Intermediate Fusion...")
try:
    fusion = IntermediateFusion(mask_missing=False)
    fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
    expected_dim = 32 + 32 + 16  # gait + non_motor + time
    print(f"Fusion: inputs {gait_emb.shape}, {non_motor_emb.shape}, {time_emb.shape} → output {fused_emb.shape}")
    print(f"Expected fused dim: {expected_dim}, Actual: {fused_emb.shape[1]}")
except Exception as e:
    print(f"Fusion failed: {e}")

print("Testing Transformer Classifier...")
try:
    transformer = TransformerClassifier(input_dim=80)  # 32+32+16
    fused_emb_seq = fused_emb.unsqueeze(1)
    output = transformer(fused_emb_seq)
    print(f"Transformer: input {fused_emb_seq.shape} → output {output.shape}")
except Exception as e:
    print(f"Transformer failed: {e}")
print("\nAll model component tests completed!")
