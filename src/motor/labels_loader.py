"""
Labels loader for motor severity prediction.
Loads clinical labels and joins with gait data by PATNO + VISNO/EVENT_ID.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def load_motor_labels(labels_path: str = "data/processed/motor_labels.csv") -> pd.DataFrame:
    """Load unified motor labels CSV."""
    return pd.read_csv(labels_path)


def map_visit_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map VISNO codes to EVENT_ID format for joining.
    Common mappings: 'bl' -> 'BL', 'V01' -> 'V01', etc.
    """
    df = df.copy()
    
    # Handle both uppercase and lowercase column names
    visno_col = None
    patno_col = None
    
    for col in df.columns:
        if col.upper() == 'VISNO':
            visno_col = col
        elif col.upper() == 'PATNO':
            patno_col = col
    
    if visno_col is not None:
        # Standardize visit codes
        df[visno_col] = df[visno_col].str.upper()
        # Map common variants
        visit_mapping = {
            'BL': 'BL',
            'V01': 'V01', 'V02': 'V02', 'V03': 'V03', 'V04': 'V04',
            'V05': 'V05', 'V06': 'V06', 'V07': 'V07', 'V08': 'V08',
            'V09': 'V09', 'V10': 'V10', 'V11': 'V11', 'V12': 'V12',
            'V13': 'V13', 'V14': 'V14', 'V15': 'V15', 'V16': 'V16',
            'V17': 'V17', 'V18': 'V18', 'V19': 'V19', 'V20': 'V20',
            'V21': 'V21', 'V22': 'V22', 'V23': 'V23', 'V24': 'V24',
            'R01': 'R01', 'R02': 'R02', 'R03': 'R03', 'R04': 'R04',
            'R05': 'R05', 'R06': 'R06', 'R07': 'R07', 'R08': 'R08',
            'R09': 'R09', 'R10': 'R10', 'R11': 'R11', 'R12': 'R12',
            'R13': 'R13', 'R14': 'R14', 'R15': 'R15', 'R16': 'R16',
            'R17': 'R17', 'R18': 'R18', 'R19': 'R19', 'R20': 'R20',
            'R21': 'R21', 'R22': 'R22', 'R23': 'R23', 'R24': 'R24',
        }
        df['EVENT_ID'] = df[visno_col].map(visit_mapping)
    
    # Ensure PATNO column exists with correct name
    if patno_col is not None and patno_col != 'PATNO':
        df['PATNO'] = df[patno_col]
    
    return df


def join_gait_with_labels(
    gait_df: pd.DataFrame, 
    labels_df: pd.DataFrame,
    modality: str = "axivity"
) -> pd.DataFrame:
    """
    Join gait data with clinical labels.
    
    Args:
        gait_df: Gait features (Axivity or Opal)
        labels_df: Clinical labels
        modality: "axivity" or "opals"
    
    Returns:
        Joined DataFrame with clinical labels
    """
    # Map visit codes in gait data
    gait_mapped = map_visit_codes(gait_df)
    
    # Filter labels by modality usage
    if modality.lower() == "axivity":
        # Only include visits where Axivity was used
        labels_filtered = labels_df[
            (labels_df['AXIVITYUSED'] == 1) | 
            (labels_df['AXIVITYUSED'].isna())
        ].copy()
    elif modality.lower() == "opals":
        # Only include visits where Opals was used
        labels_filtered = labels_df[
            (labels_df['OPALUSED'] == 1) | 
            (labels_df['OPALUSED'].isna())
        ].copy()
    else:
        labels_filtered = labels_df.copy()
    
    # Determine join columns based on what's available
    join_cols = []
    
    # Check for PATNO (case insensitive)
    patno_col = None
    for col in gait_mapped.columns:
        if col.upper() == 'PATNO':
            patno_col = col
            break
    
    if patno_col is None:
        raise ValueError("No PATNO column found in gait data")
    
    join_cols.append(patno_col)
    
    # Check for EVENT_ID
    if 'EVENT_ID' in gait_mapped.columns:
        join_cols.append('EVENT_ID')
    else:
        # If no EVENT_ID, try to use VISNO directly
        visno_col = None
        for col in gait_mapped.columns:
            if col.upper() == 'VISNO':
                visno_col = col
                break
        
        if visno_col is not None:
            # Create EVENT_ID from VISNO
            gait_mapped['EVENT_ID'] = gait_mapped[visno_col].str.upper()
            join_cols.append('EVENT_ID')
        else:
            raise ValueError("No VISNO or EVENT_ID column found in gait data")
    
    # Ensure labels_df has the same column names for joining
    labels_join_cols = ['PATNO', 'EVENT_ID']
    
    # Join on available columns
    joined = gait_mapped.merge(
        labels_filtered, 
        left_on=join_cols,
        right_on=labels_join_cols,
        how='inner'
    )
    
    return joined


def get_clinical_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract clinical targets from joined data.
    
    Returns:
        Dictionary with target arrays
    """
    targets = {}
    
    # Primary target: UPDRS Part III (motor severity)
    if 'NP3TOT' in df.columns:
        targets['updrs3'] = df['NP3TOT'].fillna(0).values
    
    # Secondary targets
    if 'NP2PTOT' in df.columns:
        targets['updrs2'] = df['NP2PTOT'].fillna(0).values
    
    if 'MSEADLG' in df.columns:
        targets['schwab_england'] = df['MSEADLG'].fillna(100).values
    
    if 'NQ_UEFS_MEAN' in df.columns:
        targets['neuroqol_ue'] = df['NQ_UEFS_MEAN'].fillna(5).values
    
    if 'NQ_LEFS_MEAN' in df.columns:
        targets['neuroqol_le'] = df['NQ_LEFS_MEAN'].fillna(5).values
    
    if 'PQUEST_SUM' in df.columns:
        targets['pquest'] = df['PQUEST_SUM'].fillna(0).values
    
    return targets


def create_regression_labels(df: pd.DataFrame, target: str = 'updrs3') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create regression labels for training.
    
    Args:
        df: Joined DataFrame with clinical labels
        target: Target variable name
    
    Returns:
        Tuple of (features, labels) arrays
    """
    # Get clinical targets
    targets = get_clinical_targets(df)
    
    if target not in targets:
        raise ValueError(f"Target '{target}' not found. Available: {list(targets.keys())}")
    
    # Extract features (exclude identifier and label columns)
    exclude_cols = [
        'PATNO', 'VISNO', 'EVENT_ID', 'PAG_NAME', 'INFODT', 'StartTime', 'StopTime',
        'NP3TOT', 'NP2PTOT', 'MSEADLG', 'NQ_UEFS_MEAN', 'NQ_LEFS_MEAN', 
        'PQUEST_SUM', 'AXIVITYUSED', 'OPALUSED', 'GAITTUG1', 'GAITTUG2',
        'patno', 'visno', 'event_id', 'pag_name', 'infodt', 'starttime', 'stoptime',
        'np3tot', 'np2ptot', 'mseadlg', 'nq_uefs_mean', 'nq_lefs_mean',
        'pquest_sum', 'axivityused', 'opalused', 'gaittug1', 'gaittug2'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features_df = df[feature_cols].select_dtypes(include=[np.number])
    
    # Get target labels
    labels = targets[target]
    
    # Remove rows with missing labels OR missing features
    valid_mask = ~np.isnan(labels)
    
    # Also check for missing features
    if len(features_df) > 0:
        feature_valid_mask = ~features_df.isnull().any(axis=1)
        valid_mask = valid_mask & feature_valid_mask
    
    features = features_df.values[valid_mask]
    labels = labels[valid_mask]
    
    # Fill any remaining NaN values with 0
    features = np.nan_to_num(features, nan=0.0)
    
    return features, labels


def get_modality_info(df: pd.DataFrame) -> Dict[str, int]:
    """Get modality usage statistics."""
    info = {}
    
    if 'AXIVITYUSED' in df.columns:
        info['axivity_visits'] = int(df['AXIVITYUSED'].sum())
    
    if 'OPALUSED' in df.columns:
        info['opals_visits'] = int(df['OPALUSED'].sum())
    
    if 'GAITTUG1' in df.columns:
        info['tug_available'] = int(df['GAITTUG1'].notna().sum())
    
    return info
