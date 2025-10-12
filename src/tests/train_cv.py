"""
Cross-validation training script for robust evaluation of clinical motor severity prediction.
Uses 5-fold CV to get more reliable performance estimates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from src.motor.axivity_data_cleaning import preprocess_axivity
from src.motor.opals_data_cleaning import preprocess_opals
from src.nonmotor.non_motor import preprocess_non_motor
from src.motor.data_fusion import (
    load_motor_labels, join_gait_with_labels, create_regression_labels, get_modality_info
)
from src.motor.gait_encoder import GaitEncoder
from src.nonmotor.non_motor_encoder import NonMotorEncoder
from src.models.time_embedding import TimeEmbedding
from src.models.intermediate_fusion import IntermediateFusion
from src.models.transform_classifier import TransformerClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_fold(train_features, train_labels, train_delta_t, val_features, val_labels, val_delta_t, 
               non_motor_features, fold_num, num_epochs=30):
    """Train a single fold."""
    
    # Initialize models
    gait_encoder = GaitEncoder(input_dim=train_features.shape[1])
    non_motor_encoder = NonMotorEncoder(input_dim=non_motor_features.shape[1])
    time_encoder = TimeEmbedding(embedding_dim=16)
    fusion = IntermediateFusion(mask_missing=False)
    transformer = TransformerClassifier(input_dim=80)
    
    # Use MSE loss for regression
    criterion = nn.MSELoss()
    
    # Optimizer with weight decay
    optimizer = optim.Adam(
        list(gait_encoder.parameters()) +
        list(non_motor_encoder.parameters()) +
        list(time_encoder.parameters()) +
        list(fusion.parameters()) +
        list(transformer.parameters()),
        lr=0.001, weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    batch_size = 8  # Smaller batch size for CV
    
    def get_batches(features, delta_t, labels, batch_size):
        for i in range(0, len(features), batch_size):
            yield (features[i:i+batch_size],
                   delta_t[i:i+batch_size],
                   labels[i:i+batch_size])
    
    # Training loop
    for epoch in range(num_epochs):
        gait_encoder.train()
        non_motor_encoder.train()
        time_encoder.train()
        fusion.train()
        transformer.train()
        
        epoch_loss = 0.0
        for batch_features, batch_delta_t, batch_labels in get_batches(train_features, train_delta_t, train_labels, batch_size):
            optimizer.zero_grad()
            
            gait_emb = gait_encoder(batch_features)
            # Use corresponding non-motor features for this batch
            batch_indices = np.random.choice(len(non_motor_features), size=len(batch_features), replace=True)
            batch_non_motor = non_motor_features[batch_indices]
            non_motor_emb = non_motor_encoder(torch.tensor(batch_non_motor, dtype=torch.float32))
            time_emb = time_encoder(batch_delta_t)
            
            fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
            fused_emb_seq = fused_emb.unsqueeze(1)
            
            output = transformer(fused_emb_seq)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_features)
        
        scheduler.step()
    
    # Validation
    gait_encoder.eval()
    non_motor_encoder.eval()
    time_encoder.eval()
    fusion.eval()
    transformer.eval()
    
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch_features, batch_delta_t, batch_labels in get_batches(val_features, val_delta_t, val_labels, batch_size):
            gait_emb = gait_encoder(batch_features)
            val_indices = np.random.choice(len(non_motor_features), size=len(batch_features), replace=True)
            val_non_motor = non_motor_features[val_indices]
            non_motor_emb = non_motor_encoder(torch.tensor(val_non_motor, dtype=torch.float32))
            time_emb = time_encoder(batch_delta_t)
            fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
            fused_emb_seq = fused_emb.unsqueeze(1)
            output = transformer(fused_emb_seq)
            
            val_predictions.extend(output.cpu().numpy().flatten())
            val_targets.extend(batch_labels.cpu().numpy().flatten())
    
    # Calculate metrics
    mse = mean_squared_error(val_targets, val_predictions)
    mae = mean_absolute_error(val_targets, val_predictions)
    r2 = r2_score(val_targets, val_predictions)
    
    print(f"Fold {fold_num}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    return mse, mae, r2

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
    
    # Use Opals as primary modality (more samples)
    primary_df = opals_labeled
    modality_name = "Opals"
    print(f"Using {modality_name} as primary modality ({len(primary_df)} samples)")
    
    # Create regression labels
    print("Creating regression labels...")
    features, labels = create_regression_labels(primary_df, target='updrs3')
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"UPDRS Part III range: {labels.min():.1f} - {labels.max():.1f}")
    
    # Load non-motor data for fusion
    try:
        print("Loading non-motor data...")
        df_non_motor = preprocess_non_motor("data/raw/non_motor/questionnaires", save=False)
        non_motor_features = df_non_motor.drop('subject_id', axis=1).select_dtypes(include='number').values
        print(f"Loaded {len(df_non_motor)} non-motor samples with {non_motor_features.shape[1]} features")
    except Exception as e:
        print(f"Could not load non-motor data: {e}")
        print("Using gait features as proxy for non-motor features")
        non_motor_features = features
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create time delta features
    delta_t = torch.rand(len(features_scaled), 1)
    
    # Convert to tensors
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    
    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    print(f"\nStarting 5-fold cross-validation with {len(features_scaled)} samples")
    
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(features_scaled)):
        print(f"\nFold {fold_num + 1}/5:")
        
        # Split data
        train_features = features_tensor[train_idx]
        train_labels = labels_tensor[train_idx]
        train_delta_t = delta_t[train_idx]
        
        val_features = features_tensor[val_idx]
        val_labels = labels_tensor[val_idx]
        val_delta_t = delta_t[val_idx]
        
        print(f"Train: {len(train_features)}, Val: {len(val_features)}")
        
        # Train fold
        mse, mae, r2 = train_fold(
            train_features, train_labels, train_delta_t,
            val_features, val_labels, val_delta_t,
            non_motor_features, fold_num + 1
        )
        
        fold_results.append({'mse': mse, 'mae': mae, 'r2': r2})
    
    # Calculate average metrics
    avg_mse = np.mean([r['mse'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    
    std_mse = np.std([r['mse'] for r in fold_results])
    std_mae = np.std([r['mae'] for r in fold_results])
    std_r2 = np.std([r['r2'] for r in fold_results])
    
    print(f"\n{'='*50}")
    print(f"5-Fold Cross-Validation Results ({modality_name})")
    print(f"{'='*50}")
    print(f"MSE: {avg_mse:.4f} ± {std_mse:.4f}")
    print(f"MAE: {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"R²:  {avg_r2:.4f} ± {std_r2:.4f}")
    print(f"{'='*50}")
    
    # Individual fold results
    print(f"\nIndividual fold results:")
    for i, result in enumerate(fold_results):
        print(f"Fold {i+1}: MSE={result['mse']:.4f}, MAE={result['mae']:.4f}, R²={result['r2']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv('cv_results.csv', index=False)
    print(f"\nCross-validation results saved to 'cv_results.csv'")

if __name__ == "__main__":
    main()
