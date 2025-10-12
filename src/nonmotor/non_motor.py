import pandas as pd
import json
import os
import glob
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def preprocess_non_motor(json_folder_path, save=True, save_path="data/processed/cleaned_non_motor_dataset.csv"):
    """
    Preprocess non-motor symptoms data from JSON questionnaire files.

    Args:
        json_folder_path: Path to folder containing JSON files
        save: Whether to save processed data
        save_path: Path to save processed CSV

    Returns:
        Processed DataFrame with non-motor features
    """

    # Load all JSON files
    json_files = glob.glob(os.path.join(json_folder_path, '*.json'))

    df_list = []
    for file in json_files:
        temp_df = pd.read_json(file)
        temp_df['subject_id'] = os.path.basename(file).replace('.json', '').replace('questionnaire_response_', '')
        df_list.append(temp_df)

    if not df_list:
        raise ValueError("No valid JSON files found")

    combined_df = pd.concat(df_list, ignore_index=True)

    # Normalize the 'item' column to separate columns
    item_expanded = pd.json_normalize(combined_df['item'])

    # Combine with subject_id
    df_expanded = pd.concat([combined_df[['subject_id']], item_expanded], axis=1)

    # Pivot to wide format
    wide_df = df_expanded.pivot(index='subject_id', columns='text', values='answer')

    # Clean up column names
    wide_df.columns.name = None
    wide_df = wide_df.reset_index()

    # Convert answers to numeric
    wide_df = wide_df.replace({True: 1, False: 0, 'Yes': 1, 'No': 0})

    # Rename columns to more readable names
    link_id_map = {
        "01": "Saliva dribbling",
        "02": "Loss taste/smell",
        "03": "Swallowing difficulty",
        "04": "Nausea",
        "05": "Constipation",
        "06": "Fecal incontinence",
        "07": "Incomplete bowel emptying",
        "08": "Urgency to urinate",
        "09": "Nocturia",
        "10": "Unexplained pains",
        "11": "Weight change",
        "12": "Memory problems",
        "13": "Loss of interest",
        "14": "Hallucinations",
        "15": "Concentration difficulty",
        "16": "Feeling sad",
        "17": "Feeling anxious",
        "18": "Sex interest change",
        "19": "Sexual difficulty",
        "20": "Dizziness on standing",
        "21": "Falls",
        "22": "Daytime sleepiness",
        "23": "Sleep problems",
        "24": "Vivid/frightening dreams",
        "25": "Dream enactment",
        "26": "Restless legs",
        "27": "Leg swelling",
        "28": "Excess sweating",
        "29": "Double vision",
        "30": "Delusions or paranoia"
    }

    # Rename columns
    wide_df = wide_df.rename(columns=link_id_map)

    # Remove duplicates
    wide_df = wide_df.drop_duplicates()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        wide_df.to_csv(save_path, index=False)
        print(f"Non-motor data saved to {save_path}")

    return wide_df

def load_patient_labels(json_folder_path):
    """
    Load patient condition labels from JSON files.

    Args:
        json_folder_path: Path to folder containing patient JSON files

    Returns:
        DataFrame with patient labels
    """
    json_files = glob.glob(os.path.join(json_folder_path, "*.json"))

    patient_data = []
    for file in json_files:
        with open(file, "r", encoding='utf-8') as f:
            data = json.load(f)

        patient_id = data.get("id")
        condition_raw = str(data.get("condition", "")).lower()

        condition = 0 if "healthy" in condition_raw else 1

        patient_data.append({
            "subject_id": patient_id,
            "condition": condition,
            "gender": data.get("gender"),
            "appearance_in_kinship": data.get("appearance_in_kinship")
        })

    df_patients = pd.DataFrame(patient_data)
    return df_patients

def preprocess_non_motor_with_labels(questionnaire_folder, patients_folder,
                                   save=True, save_path="data/processed/cleaned_non_motor_with_labels.csv"):
    """
    Preprocess non-motor data and merge with patient labels.

    Args:
        questionnaire_folder: Path to questionnaire JSON files
        patients_folder: Path to patient JSON files
        save: Whether to save processed data
        save_path: Path to save processed CSV

    Returns:
        Processed DataFrame with features and labels
    """

    # Load and preprocess questionnaire data
    df_questionnaire = preprocess_non_motor(questionnaire_folder, save=False)

    # Load patient labels
    df_patients = load_patient_labels(patients_folder)

    # Merge data
    result = df_questionnaire.merge(df_patients, on="subject_id", how="left")

    # Handle missing values
    result = result.fillna(0)  # Fill NaN with 0 for symptoms

    # Encode categorical features
    X = result.drop(['condition', 'subject_id'], axis=1, errors='ignore')
    y = result['condition']

    X_encoded = X.copy()
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

    # Apply SMOTE for class balance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

    # Convert back to DataFrame
    X_resampled = pd.DataFrame(X_resampled, columns=X_encoded.columns)
    y_resampled = pd.Series(y_resampled, name='condition')

    balanced_df = pd.concat([X_resampled, y_resampled], axis=1)

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        balanced_df.to_csv(save_path, index=False)
        print(f"Balanced non-motor data saved to {save_path}")

    return balanced_df
