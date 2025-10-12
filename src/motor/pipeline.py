import torch
from src.motor.preprocess_axivity import preprocess_axivity
from src.motor.preprocess_opals import preprocess_opals
from src.motor.gait_encoder import GaitEncoder
from src.motor.utils import save_embeddings

def generate_motor_embeddings(axivity_path, opals_path, save_path="embeddings/motor_embeddings.pth"):
    """
    Generate and save motor embeddings (Axivity + Opals) without temporal or multimodal fusion.
    """
    df_axivity = preprocess_axivity(axivity_path, save=False)
    df_opals = preprocess_opals(opals_path, save=False)

    encoder_ax = GaitEncoder(input_dim=df_axivity.shape[1])
    encoder_op = GaitEncoder(input_dim=df_opals.shape[1])
    ax_emb = encoder_ax(torch.tensor(df_axivity.values, dtype=torch.float32))
    op_emb = encoder_op(torch.tensor(df_opals.values, dtype=torch.float32))

    motor_emb = torch.cat([ax_emb, op_emb], dim=1)
    save_embeddings(motor_emb, save_path)
    print(f"Motor embeddings saved at {save_path}")

    return motor_emb

if __name__ == "__main__":
    generate_motor_embeddings("data/raw/axivity.csv", "data/raw/opals.csv")
