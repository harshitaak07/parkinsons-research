"""
Multi-target training script for clinical motor severity prediction.
Predicts UPDRS Part III (primary) + UPDRS Part II + Schwab & England + NeuroQoL scores.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from src.motor.axivity_data_cleaning import preprocess_axivity
from src.motor.opals_data_cleaning import preprocess_opals
from src.nonmotor.non_motor import preprocess_non_motor
from src.motor.data_fusion import (
    load_motor_labels, join_gait_with_labels, get_clinical_targets, get_modality_info
)
from src.motor.gait_encoder import GaitEncoder
from src.nonmotor.non_motor_encoder import NonMotorEncoder
from src.models.time_embedding import TimeEmbedding
from src.models.intermediate_fusion import IntermediateFusion
from src.models.transform_classifier import TransformerClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MultiTargetTransformerClassifier(nn.Module):
    """Transformer classifier for multi-target regression."""
    
    def __init__(self, input_dim, num_targets=4, hidden_dim=128, num_heads=8, num_layers=3):
        super().__init__()
        self.num_targets = num_targets
        
        # Shared transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Separate heads for each target
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_targets)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        transformer_out = self.transformer(x)
        
        # Use the last timestep for prediction
        last_output = transformer_out[:, -1, :]  # (batch_size, input_dim)
        
        # Generate predictions for each target
        predictions = []
        for head in self.heads:
            pred = head(last_output)
            predictions.append(pred)
        
        return torch.cat(predictions, dim=1)  # (batch_size, num_targets)

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
    
    # Choose primary modality for training
    if len(axivity_labeled) > 0:
        primary_df = axivity_labeled
        modality_name = "Axivity"
        print(f"Using {modality_name} as primary modality ({len(primary_df)} samples)")
    elif len(opals_labeled) > 0:
        primary_df = opals_labeled
        modality_name = "Opals"
        print(f"Using {modality_name} as primary modality ({len(primary_df)} samples)")
    else:
        raise ValueError("No labeled gait data available for training")
    
    # Extract features and multiple targets
    print("Creating multi-target labels...")
    
    # Extract features (exclude identifier and label columns)
    exclude_cols = [
        'PATNO', 'VISNO', 'EVENT_ID', 'PAG_NAME', 'INFODT', 'StartTime', 'StopTime',
        'NP3TOT', 'NP2PTOT', 'MSEADLG', 'NQ_UEFS_MEAN', 'NQ_LEFS_MEAN', 
        'PQUEST_SUM', 'AXIVITYUSED', 'OPALUSED', 'GAITTUG1', 'GAITTUG2',
        'patno', 'visno', 'event_id', 'pag_name', 'infodt', 'starttime', 'stoptime',
        'np3tot', 'np2ptot', 'mseadlg', 'nq_uefs_mean', 'nq_lefs_mean',
        'pquest_sum', 'axivityused', 'opalused', 'gaittug1', 'gaittug2'
    ]
    
    feature_cols = [col for col in primary_df.columns if col not in exclude_cols]
    features_df = primary_df[feature_cols].select_dtypes(include=[np.number])
    
    # Get multiple targets
    targets = get_clinical_targets(primary_df)
    target_names = ['updrs3', 'updrs2', 'schwab_england', 'neuroqol_ue']
    available_targets = [name for name in target_names if name in targets]
    
    print(f"Available targets: {available_targets}")
    
    # Create target matrix
    target_matrix = np.column_stack([targets[name] for name in available_targets])
    
    # Remove rows with missing data
    valid_mask = ~np.isnan(target_matrix).any(axis=1)
    if len(features_df) > 0:
        feature_valid_mask = ~features_df.isnull().any(axis=1)
        valid_mask = valid_mask & feature_valid_mask
    
    features = features_df.values[valid_mask]
    target_matrix = target_matrix[valid_mask]
    
    # Fill any remaining NaN values
    features = np.nan_to_num(features, nan=0.0)
    target_matrix = np.nan_to_num(target_matrix, nan=0.0)
    
    print(f"Features shape: {features.shape}, Targets shape: {target_matrix.shape}")
    print(f"Target ranges:")
    for i, name in enumerate(available_targets):
        print(f"  {name}: {target_matrix[:, i].min():.1f} - {target_matrix[:, i].max():.1f}")
    
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
    targets_tensor = torch.tensor(target_matrix, dtype=torch.float32)
    
    # Initialize models
    gait_encoder = GaitEncoder(input_dim=features_scaled.shape[1])
    non_motor_encoder = NonMotorEncoder(input_dim=non_motor_features.shape[1])
    time_encoder = TimeEmbedding(embedding_dim=16)
    fusion = IntermediateFusion(mask_missing=False)
    transformer = MultiTargetTransformerClassifier(
        input_dim=80,  # 32 (gait) + 32 (non-motor) + 16 (time)
        num_targets=len(available_targets)
    )
    
    # Use MSE loss for multi-target regression
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
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training parameters
    num_epochs = 50
    batch_size = 16
    
    # Split data into train/validation
    train_size = int(0.8 * len(features_scaled))
    indices = np.random.permutation(len(features_scaled))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_features = features_tensor[train_indices]
    train_targets = targets_tensor[train_indices]
    train_delta_t = delta_t[train_indices]
    
    val_features = features_tensor[val_indices]
    val_targets = targets_tensor[val_indices]
    val_delta_t = delta_t[val_indices]
    
    def get_batches(features, delta_t, targets, batch_size):
        for i in range(0, len(features), batch_size):
            yield (features[i:i+batch_size],
                   delta_t[i:i+batch_size],
                   targets[i:i+batch_size])
    
    print(f"Starting multi-target training with {len(train_features)} train samples, {len(val_features)} val samples")
    
    # Training loop
    for epoch in range(num_epochs):
        gait_encoder.train()
        non_motor_encoder.train()
        time_encoder.train()
        fusion.train()
        transformer.train()
        
        epoch_loss = 0.0
        for batch_features, batch_delta_t, batch_targets in get_batches(train_features, train_delta_t, train_targets, batch_size):
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
            loss = criterion(output, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_features)
        
        avg_loss = epoch_loss / len(train_features)
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            gait_encoder.eval()
            non_motor_encoder.eval()
            time_encoder.eval()
            fusion.eval()
            transformer.eval()
            
            val_loss = 0.0
            val_predictions = []
            val_targets_list = []
            
            with torch.no_grad():
                for batch_features, batch_delta_t, batch_targets in get_batches(val_features, val_delta_t, val_targets, batch_size):
                    gait_emb = gait_encoder(batch_features)
                    val_indices = np.random.choice(len(non_motor_features), size=len(batch_features), replace=True)
                    val_non_motor = non_motor_features[val_indices]
                    non_motor_emb = non_motor_encoder(torch.tensor(val_non_motor, dtype=torch.float32))
                    time_emb = time_encoder(batch_delta_t)
                    fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
                    fused_emb_seq = fused_emb.unsqueeze(1)
                    output = transformer(fused_emb_seq)
                    loss = criterion(output, batch_targets)
                    val_loss += loss.item() * len(batch_features)
                    
                    val_predictions.extend(output.cpu().numpy())
                    val_targets_list.extend(batch_targets.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_features)
            
            # Calculate regression metrics for each target
            val_predictions = np.array(val_predictions)
            val_targets_array = np.array(val_targets_list)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            for i, target_name in enumerate(available_targets):
                mse = mean_squared_error(val_targets_array[:, i], val_predictions[:, i])
                mae = mean_absolute_error(val_targets_array[:, i], val_predictions[:, i])
                r2 = r2_score(val_targets_array[:, i], val_predictions[:, i])
                print(f"  {target_name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
        
        scheduler.step()
    
    print("Multi-target training complete.")
    
    # Final evaluation
    gait_encoder.eval()
    non_motor_encoder.eval()
    time_encoder.eval()
    fusion.eval()
    transformer.eval()
    
    final_predictions = []
    final_targets = []
    
    with torch.no_grad():
        for batch_features, batch_delta_t, batch_targets in get_batches(val_features, val_delta_t, val_targets, batch_size):
            gait_emb = gait_encoder(batch_features)
            val_indices = np.random.choice(len(non_motor_features), size=len(batch_features), replace=True)
            val_non_motor = non_motor_features[val_indices]
            non_motor_emb = non_motor_encoder(torch.tensor(val_non_motor, dtype=torch.float32))
            time_emb = time_encoder(batch_delta_t)
            fused_emb = fusion(gait_emb, non_motor_emb, time_emb)
            fused_emb_seq = fused_emb.unsqueeze(1)
            output = transformer(fused_emb_seq)
            
            final_predictions.extend(output.cpu().numpy())
            final_targets.extend(batch_targets.cpu().numpy())
    
    # Final metrics
    final_predictions = np.array(final_predictions)
    final_targets = np.array(final_targets)
    
    print(f"\nFinal Multi-Target Validation Results:")
    for i, target_name in enumerate(available_targets):
        mse = mean_squared_error(final_targets[:, i], final_predictions[:, i])
        mae = mean_absolute_error(final_targets[:, i], final_predictions[:, i])
        r2 = r2_score(final_targets[:, i], final_predictions[:, i])
        print(f"{target_name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    
    # Save model
    torch.save({
        'gait_encoder': gait_encoder.state_dict(),
        'non_motor_encoder': non_motor_encoder.state_dict(),
        'time_encoder': time_encoder.state_dict(),
        'fusion': fusion.state_dict(),
        'transformer': transformer.state_dict(),
        'scaler': scaler,
        'modality': modality_name,
        'target_names': available_targets,
        'final_metrics': {name: {
            'mse': mean_squared_error(final_targets[:, i], final_predictions[:, i]),
            'mae': mean_absolute_error(final_targets[:, i], final_predictions[:, i]),
            'r2': r2_score(final_targets[:, i], final_predictions[:, i])
        } for i, name in enumerate(available_targets)}
    }, 'multitarget_model.pth')
    
    print(f"Multi-target model saved as 'multitarget_model.pth'")

if __name__ == "__main__":
    main()
