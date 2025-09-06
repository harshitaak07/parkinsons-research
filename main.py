from preprocessing.axivity_gait import preprocess_axivity
from preprocessing.opals_gait import preprocess_opals

df_axivity = preprocess_axivity("data/raw/Gait_Data___Arm_swing__Axivity__06Sep2025.csv")
df_opals   = preprocess_opals("data/raw/Gait_Data___Arm_swing__Opals__07Aug2025.csv")