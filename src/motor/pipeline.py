import torch
from src.motor.axivity_data_cleaning import preprocess_axivity
from src.motor.opals_data_cleaning import preprocess_opals
from src.motor.gait_encoder import GaitEncoder
from src.motor.utils import save_embeddings

def generate_motor_embeddings(axivity_path, opals_path, save_path="embeddings/motor_embeddings.pth"):
    """
    Generate and save motor embeddings (Axivity + Opals) without temporal or multimodal fusion.
    """
    df_axivity = preprocess_axivity(axivity_path, save=False)
    df_opals = preprocess_opals(opals_path, save=False)

    # 🔧 Select only numeric features
    df_axivity_num = df_axivity.select_dtypes(include=["number"])
    df_opals_num = df_opals.select_dtypes(include=["number"])

    # 🔥 Initialize encoders
    encoder_ax = GaitEncoder(input_dim=df_axivity_num.shape[1])
    encoder_op = GaitEncoder(input_dim=df_opals_num.shape[1])

    # 🧠 Compute embeddings
    ax_emb = encoder_ax(torch.tensor(df_axivity_num.values, dtype=torch.float32))
    op_emb = encoder_op(torch.tensor(df_opals_num.values, dtype=torch.float32))

    # 🧩 Align sizes before concatenation (if necessary)
    min_len = min(ax_emb.shape[0], op_emb.shape[0])
    ax_emb, op_emb = ax_emb[:min_len], op_emb[:min_len]

    motor_emb = torch.cat([ax_emb, op_emb], dim=1)
    save_embeddings(motor_emb, save_path)
    print(f"✅ Motor embeddings saved at {save_path}")

    return motor_emb


if __name__ == "__main__":
    generate_motor_embeddings(
        "data/raw/motor/Gait_Data___Arm_swing__Axivity__06Sep2025.csv",
        "data/raw/motor/Gait_Data___Arm_swing__Opals__07Aug2025.csv"
    )
