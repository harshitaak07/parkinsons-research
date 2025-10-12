"""
Hyperparameter-tuned training script for clinical motor severity prediction.
Uses the most important features identified in feature analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from motor.axivity_gait import preprocess_axivity
from motor.opals_gait import preprocess_opals
from nonmotor.non_motor import preprocess_non_motor
from motor.labels_loader import (
    load_motor_labels, join_gait_with_labels, create_regression_labels, get_modality_info
)
from motor.gait_encoder import GaitEncoder
from nonmotor.non_motor_encoder import NonMotorEncoder
from models.time_embedding import TimeEmbedding
from models.intermediate_fusion import IntermediateFusion
from src.models.transform_classifier import TransformerClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_top_features(df, target_col='NP3TOT', top_n=10):
    """Get top N most important features based on correlation with target."""
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target and identifier columns
    exclude_cols = [
        'PATNO', 'VISNO', 'EVENT_ID', 'PAG_NAME', 'INFODT', 'StartTime', 'StopTime',
        'NP3TOT', 'NP2PTOT', 'MSEADLG', 'NQ_UEFS_MEAN', 'NQ_LEFS_MEAN', 
        'PQUEST_SUM', 'AXIVITYUSED', 'OPALUSED', 'GAITTUG1', 'GAITTUG2',
        'patno', 'visno', 'event_id', 'pag_name', 'infodt', 'starttime', 'stoptime',
        'np3tot', 'np2ptot', 'mseadlg', 'nq_uefs_mean', 'nq_lefs_mean',
        'pquest_sum', 'axivityused', 'opalused', 'gaittug1', 'gaittug2'
    ]
    
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlations with target
    correlations = []
    for col in feature_cols:
        if col in df.columns and target_col in df.columns:
            corr = df[col].corr(df[target_col])
            if not np.isnan(corr):
                correlations.append((col, abs(corr)))
    
    # Sort by correlation and get top N
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [col for col, _ in correlations[:top_n]]
    
    return top_features

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
    
    # Get top features based on correlation with UPDRS Part III
    print("Identifying top features...")
    top_features = get_top_features(primary_df, target_col='NP3TOT', top_n=15)
    print(f"Top {len(top_features)} features: {top_features}")
    
    # Create regression labels
    print("Creating regression labels...")
    features, labels = create_regression_labels(primary_df, target='updrs3')
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"UPDRS Part III range: {labels.min():.1f} - {labels.max():.1f}")
    
    # Filter features to only include top features
    if len(top_features) > 0:
        # Get feature names from the original dataframe
        exclude_cols = [
            'PATNO', 'VISNO', 'EVENT_ID', 'PAG_NAME', 'INFODT', 'StartTime', 'StopTime',
            'NP3TOT', 'NP2PTOT', 'MSEADLG', 'NQ_UEFS_MEAN', 'NQ_LEFS_MEAN', 
            'PQUEST_SUM', 'AXIVITYUSED', 'OPALUSED', 'GAITTUG1', 'GAITTUG2',
            'patno', 'visno', 'event_id', 'pag_name', 'infodt', 'starttime', 'stoptime',
            'np3tot', 'np2ptot', 'mseadlg', 'nq_uefs_mean', 'nq_lefs_mean',
            'pquest_sum', 'axivityused', 'opalused', 'gaittug1', 'gaittug2'
        ]
        
        feature_cols = [col for col in primary_df.columns if col not in exclude_cols]
        feature_names = [col for col in feature_cols if col in primary_df.select_dtypes(include=[np.number]).columns]
        
        # Find indices of top features
        top_feature_indices = []
        for feature in top_features:
            if feature in feature_names:
                top_feature_indices.append(feature_names.index(feature))
        
        if top_feature_indices:
            features = features[:, top_feature_indices]
            print(f"Filtered to top {len(top_feature_indices)} features: {[feature_names[i] for i in top_feature_indices]}")
    
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
    
    # Initialize models with tuned hyperparameters
    gait_encoder = GaitEncoder(input_dim=features_scaled.shape[1])
    non_motor_encoder = NonMotorEncoder(input_dim=non_motor_features.shape[1])
    time_encoder = TimeEmbedding(embedding_dim=32)  # Increased embedding dimension
    fusion = IntermediateFusion(mask_missing=False)
    transformer = TransformerClassifier(input_dim=96)  # 32 (gait) + 32 (non-motor) + 32 (time)
    
    # Use MSE loss for regression
    criterion = nn.MSELoss()
    
    # Optimizer with tuned hyperparameters
    optimizer = optim.Adam(
        list(gait_encoder.parameters()) +
        list(non_motor_encoder.parameters()) +
        list(time_encoder.parameters()) +
        list(fusion.parameters()) +
        list(transformer.parameters()),
        lr=0.0005, weight_decay=1e-3  # Lower learning rate, higher weight decay
    )
    
    # Learning rate scheduler with different parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training parameters
    num_epochs = 100  # More epochs
    batch_size = 8   # Smaller batch size
    
    # Split data into train/validation
    train_size = int(0.8 * len(features_scaled))
    indices = np.random.permutation(len(features_scaled))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_features = features_tensor[train_indices]
    train_labels = labels_tensor[train_indices]
    train_delta_t = delta_t[train_indices]
    
    val_features = features_tensor[val_indices]
    val_labels = labels_tensor[val_indices]
    val_delta_t = delta_t[val_indices]
    
    def get_batches(features, delta_t, labels, batch_size):
        for i in range(0, len(features), batch_size):
            yield (features[i:i+batch_size],
                   delta_t[i:i+batch_size],
                   labels[i:i+batch_size])
    
    print(f"Starting tuned training with {len(train_features)} train samples, {len(val_features)} val samples")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(gait_encoder.parameters()) +
                list(non_motor_encoder.parameters()) +
                list(time_encoder.parameters()) +
                list(fusion.parameters()) +
                list(transformer.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_features)
        
        avg_loss = epoch_loss / len(train_features)
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            gait_encoder.eval()
            non_motor_encoder.eval()
            time_encoder.eval()
            fusion.eval()
            transformer.eval()
            
            val_loss = 0.0
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
                    loss = criterion(output, batch_labels)
                    val_loss += loss.item() * len(batch_features)
                    
                    val_predictions.extend(output.cpu().numpy().flatten())
                    val_targets.extend(batch_labels.cpu().numpy().flatten())
            
            avg_val_loss = val_loss / len(val_features)
            
            # Calculate regression metrics
            mse = mean_squared_error(val_targets, val_predictions)
            mae = mean_absolute_error(val_targets, val_predictions)
            r2 = r2_score(val_targets, val_predictions)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"  Val MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'gait_encoder': gait_encoder.state_dict(),
                    'non_motor_encoder': non_motor_encoder.state_dict(),
                    'time_encoder': time_encoder.state_dict(),
                    'fusion': fusion.state_dict(),
                    'transformer': transformer.state_dict(),
                    'scaler': scaler,
                    'modality': modality_name,
                    'top_features': top_features,
                    'metrics': {'mse': mse, 'mae': mae, 'r2': r2}
                }, 'best_tuned_model.pth')
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
    
    print("Tuned training complete.")
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_tuned_model.pth', weights_only=False)
    gait_encoder.load_state_dict(checkpoint['gait_encoder'])
    non_motor_encoder.load_state_dict(checkpoint['non_motor_encoder'])
    time_encoder.load_state_dict(checkpoint['time_encoder'])
    fusion.load_state_dict(checkpoint['fusion'])
    transformer.load_state_dict(checkpoint['transformer'])
    
    # Final evaluation
    gait_encoder.eval()
    non_motor_encoder.eval()
    time_encoder.eval()
    fusion.eval()
    transformer.eval()
    
    final_predictions = []
    final_targets = []
    
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
            
            final_predictions.extend(output.cpu().numpy().flatten())
            final_targets.extend(batch_labels.cpu().numpy().flatten())
    
    # Final metrics
    final_mse = mean_squared_error(final_targets, final_predictions)
    final_mae = mean_absolute_error(final_targets, final_predictions)
    final_r2 = r2_score(final_targets, final_predictions)
    
    print(f"\nFinal Tuned Validation Results:")
    print(f"MSE: {final_mse:.4f}")
    print(f"MAE: {final_mae:.4f}")
    print(f"R²: {final_r2:.4f}")
    print(f"Top features used: {top_features}")
    
    print(f"Best tuned model saved as 'best_tuned_model.pth'")

if __name__ == "__main__":
    main()
