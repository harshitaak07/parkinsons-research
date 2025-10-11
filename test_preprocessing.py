import pandas as pd
from src.preprocessing.axivity_gait import preprocess_axivity
from src.preprocessing.opals_gait import preprocess_opals
from src.preprocessing.non_motor import preprocess_non_motor

print("Testing Axivity preprocessing...")
try:
    df_axivity = preprocess_axivity("data/raw/motor/Gait_Data___Arm_swing__Axivity__06Sep2025.csv", save=False)
    print(f"✅ Axivity preprocessing successful: {df_axivity.shape[0]} samples, {df_axivity.shape[1]} features")
except Exception as e:
    print(f"❌ Axivity preprocessing failed: {e}")

print("\nTesting Opals preprocessing...")
try:
    df_opals = preprocess_opals("data/raw/motor/Gait_Data___Arm_swing__Opals__07Aug2025.csv", save=False)
    print(f"✅ Opals preprocessing successful: {df_opals.shape[0]} samples, {df_opals.shape[1]} features")
except Exception as e:
    print(f"❌ Opals preprocessing failed: {e}")

print("\nTesting Non-motor preprocessing...")
try:
    df_non_motor = preprocess_non_motor("data/raw/non_motor/questionnaires", save=False)
    print(f"✅ Non-motor preprocessing successful: {df_non_motor.shape[0]} samples, {df_non_motor.shape[1]} features")
    print(f"   Features: {list(df_non_motor.columns)}")
except Exception as e:
    print(f"❌ Non-motor preprocessing failed: {e}")

print("\nAll preprocessing tests completed!")
