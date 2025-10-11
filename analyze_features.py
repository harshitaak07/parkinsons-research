"""
Feature importance analysis for clinical motor severity prediction.
Analyzes which gait features are most predictive of UPDRS Part III scores.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from src.preprocessing.axivity_gait import preprocess_axivity
from src.preprocessing.opals_gait import preprocess_opals
from src.preprocessing.non_motor import preprocess_non_motor
from src.preprocessing.labels_loader import (
    load_motor_labels, join_gait_with_labels, create_regression_labels
)
# import matplotlib.pyplot as plt
# import seaborn as sns

def analyze_feature_importance(features, labels, feature_names, method='random_forest'):
    """Analyze feature importance using different methods."""
    
    if method == 'random_forest':
        # Random Forest feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df, rf
    
    elif method == 'permutation':
        # Permutation importance
        from sklearn.inspection import permutation_importance
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        
        perm_importance = permutation_importance(rf, features, labels, n_repeats=10, random_state=42)
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df, rf
    
    else:
        raise ValueError("Method must be 'random_forest' or 'permutation'")

def print_feature_importance(feature_importance_df, top_n=20, title="Feature Importance"):
    """Print feature importance."""
    
    print(f"\n{title} (Top {top_n}):")
    print("-" * 50)
    
    # Get top N features
    top_features = feature_importance_df.head(top_n)
    
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    print("-" * 50)

def analyze_feature_correlations(features, labels, feature_names):
    """Analyze correlations between features and target."""
    
    # Calculate correlations
    correlations = []
    for i, feature_name in enumerate(feature_names):
        corr = np.corrcoef(features[:, i], labels)[0, 1]
        correlations.append(abs(corr))  # Use absolute correlation
    
    # Create correlation DataFrame
    corr_df = pd.DataFrame({
        'feature': feature_names,
        'correlation': correlations
    }).sort_values('correlation', ascending=False)
    
    return corr_df

def main():
    print("Loading clinical motor labels...")
    labels_df = load_motor_labels()
    print(f"Loaded {len(labels_df)} clinical records")
    
    # Load and preprocess gait data
    print("Loading Axivity gait data...")
    df_axivity = preprocess_axivity("data/raw/motor/Gait_Data___Arm_swing__Axivity__06Sep2025.csv", save=False)
    print(f"Loaded {len(df_axivity)} Axivity records")
    
    print("Loading Opals gait data...")
    df_opals = preprocess_opals("data/raw/motor/Gait_Data___Arm_swing__Opals__07Aug2025.csv", save=False)
    print(f"Loaded {len(df_opals)} Opals records")
    
    # Join gait data with clinical labels
    print("Joining Axivity data with clinical labels...")
    axivity_labeled = join_gait_with_labels(df_axivity, labels_df, modality="axivity")
    print(f"Joined {len(axivity_labeled)} Axivity records with labels")
    
    print("Joining Opals data with clinical labels...")
    opals_labeled = join_gait_with_labels(df_opals, labels_df, modality="opals")
    print(f"Joined {len(opals_labeled)} Opals records with labels")
    
    # Analyze both modalities
    modalities = [
        ("Axivity", axivity_labeled),
        ("Opals", opals_labeled)
    ]
    
    for modality_name, df in modalities:
        if len(df) == 0:
            continue
            
        print(f"\n{'='*60}")
        print(f"Feature Analysis: {modality_name}")
        print(f"{'='*60}")
        
        # Create regression labels
        features, labels = create_regression_labels(df, target='updrs3')
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        
        if len(features) == 0:
            print("No valid samples found")
            continue
        
        # Get feature names
        exclude_cols = [
            'PATNO', 'VISNO', 'EVENT_ID', 'PAG_NAME', 'INFODT', 'StartTime', 'StopTime',
            'NP3TOT', 'NP2PTOT', 'MSEADLG', 'NQ_UEFS_MEAN', 'NQ_LEFS_MEAN', 
            'PQUEST_SUM', 'AXIVITYUSED', 'OPALUSED', 'GAITTUG1', 'GAITTUG2',
            'patno', 'visno', 'event_id', 'pag_name', 'infodt', 'starttime', 'stoptime',
            'np3tot', 'np2ptot', 'mseadlg', 'nq_uefs_mean', 'nq_lefs_mean',
            'pquest_sum', 'axivityused', 'opalused', 'gaittug1', 'gaittug2'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_names = [col for col in feature_cols if col in df.select_dtypes(include=[np.number]).columns]
        
        # Ensure we have the right number of feature names
        if len(feature_names) != features.shape[1]:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        print(f"Analyzing {len(feature_names)} features")
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 1. Random Forest Feature Importance
        print("\n1. Random Forest Feature Importance:")
        rf_importance_df, rf_model = analyze_feature_importance(
            features_scaled, labels, feature_names, method='random_forest'
        )
        
        print("Top 10 most important features:")
        print(rf_importance_df.head(10))
        
        # 2. Permutation Importance
        print("\n2. Permutation Importance:")
        perm_importance_df, _ = analyze_feature_importance(
            features_scaled, labels, feature_names, method='permutation'
        )
        
        print("Top 10 most important features (permutation):")
        print(perm_importance_df.head(10))
        
        # 3. Correlation Analysis
        print("\n3. Correlation Analysis:")
        corr_df = analyze_feature_correlations(features_scaled, labels, feature_names)
        
        print("Top 10 most correlated features:")
        print(corr_df.head(10))
        
        # 4. Model Performance with Top Features
        print("\n4. Model Performance with Top Features:")
        
        # Test different numbers of top features
        for n_features in [5, 10, 15, 20]:
            if n_features > len(feature_names):
                continue
                
            # Get top N features
            top_features_idx = rf_importance_df.head(n_features).index
            top_features = features_scaled[:, top_features_idx]
            
            # Train Random Forest with top features
            rf_top = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_top.fit(top_features, labels)
            
            # Predictions
            predictions = rf_top.predict(top_features)
            
            # Metrics
            mse = mean_squared_error(labels, predictions)
            r2 = r2_score(labels, predictions)
            
            print(f"Top {n_features} features: MSE={mse:.4f}, R²={r2:.4f}")
        
        # 5. Feature Groups Analysis (for Opals)
        if modality_name == "Opals":
            print("\n5. Feature Groups Analysis (Opals):")
            
            # Define feature groups based on Opals preprocessing
            feature_groups = {
                'Walking Speed and Cadence': ['SP_U', 'CAD_U'],
                'Arm Swing Amplitude and Variability': ['RA_AMP_U', 'LA_AMP_U', 'RA_STD_U', 'LA_STD_U'],
                'Symmetry and Asymmetry Measures': ['SYM_U', 'ASYM_IND_U'],
                'Stride and Step Timing and Regularity': ['STR_T_U', 'STR_CV_U', 'STEP_REG_U', 'STEP_SYM_U'],
                'Movement Smoothness / Jerk Measures': ['JERK_T_U', 'R_JERK_U', 'L_JERK_U'],
                'Functional Mobility (TUG) Test Metrics': ['TUG1_DUR', 'TUG1_STEP_NUM', 'TUG1_STRAIGHT_DUR', 'TUG1_TURNS_DUR', 'TUG1_STEP_REG', 'TUG1_STEP_SYM']
            }
            
            # Calculate group importance
            group_importance = {}
            for group_name, group_features in feature_groups.items():
                # Find features in this group
                group_idx = []
                for feature in group_features:
                    if feature in feature_names:
                        group_idx.append(feature_names.index(feature))
                
                if group_idx:
                    # Calculate average importance for this group
                    group_importance[group_name] = rf_importance_df.iloc[group_idx]['importance'].mean()
            
            # Sort by importance
            group_importance_df = pd.DataFrame([
                {'group': group, 'importance': importance}
                for group, importance in group_importance.items()
            ]).sort_values('importance', ascending=False)
            
            print("Feature group importance:")
            print(group_importance_df)
        
        # Save results
        rf_importance_df.to_csv(f'{modality_name.lower()}_feature_importance.csv', index=False)
        perm_importance_df.to_csv(f'{modality_name.lower()}_permutation_importance.csv', index=False)
        corr_df.to_csv(f'{modality_name.lower()}_correlations.csv', index=False)
        
        print(f"\nResults saved for {modality_name}")
    
    print(f"\n{'='*60}")
    print("Feature analysis complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
