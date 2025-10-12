import torch
import os
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

print("Testing Model Loading...")

gait_dim = 103
non_motor_dim = 30
gait_encoder = GaitEncoder(input_dim=gait_dim)
non_motor_encoder = NonMotorEncoder(input_dim=non_motor_dim)
time_encoder = TimeEmbedding(embedding_dim=16)
fusion = IntermediateFusion(mask_missing=False)
transformer = TransformerClassifier(input_dim=80)

model_files = {
    'gait_encoder': 'gait_encoder.pth',
    'non_motor_encoder': 'non_motor_encoder.pth',
    'time_encoder': 'time_encoder.pth',
    'fusion': 'fusion.pth',
    'transformer': 'transformer.pth'
}

loaded_models = {}
for model_name, file_path in model_files.items():
    if os.path.exists(file_path):
        try:
            if model_name == 'gait_encoder':
                gait_encoder.load_state_dict(torch.load(file_path))
                loaded_models[model_name] = gait_encoder
            elif model_name == 'non_motor_encoder':
                non_motor_encoder.load_state_dict(torch.load(file_path))
                loaded_models[model_name] = non_motor_encoder
            elif model_name == 'time_encoder':
                time_encoder.load_state_dict(torch.load(file_path))
                loaded_models[model_name] = time_encoder
            elif model_name == 'fusion':
                fusion.load_state_dict(torch.load(file_path))
                loaded_models[model_name] = fusion
            elif model_name == 'transformer':
                transformer.load_state_dict(torch.load(file_path))
                loaded_models[model_name] = transformer
            print(f"{model_name} loaded successfully from {file_path}")
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
    else:
        print(f"{file_path} not found")
print(f"\nLoaded {len(loaded_models)}/{len(model_files)} models")

if len(loaded_models) == len(model_files):
    print("\nTesting inference with loaded models...")
    try:
        batch_size = 5
        gait_input = torch.randn(batch_size, gait_dim)
        non_motor_input = torch.randn(batch_size, non_motor_dim)
        time_input = torch.randn(batch_size, 1)
        gait_encoder.eval()
        non_motor_encoder.eval()
        time_encoder.eval()
        fusion.eval()
        transformer.eval()
        with torch.no_grad():
            gait_emb = gait_encoder(gait_input)
            non_motor_emb = non_motor_encoder(non_motor_input)
            time_emb = time_encoder(time_input)
            fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
            fused_emb_seq = fused_emb.unsqueeze(1)
            output = transformer(fused_emb_seq)
        print(f"Inference successful: {output.shape}")
        print(f"Sample outputs: {output.squeeze().tolist()[:3]}")
    except Exception as e:
        print(f"Inference failed: {e}")
else:
    print("Cannot test inference - not all models loaded")

print("\nModel loading tests completed!")
