"""
Training script using Opals gait data (282 samples) for clinical motor severity prediction.
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
    
    # Print modality usage info
    opals_info = get_modality_info(opals_labeled)
    print(f"Opals info: {opals_info}")
    
    # Create regression labels (UPDRS Part III as primary target)
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
    
    # Initialize models
    gait_encoder = GaitEncoder(input_dim=features_scaled.shape[1])
    non_motor_encoder = NonMotorEncoder(input_dim=non_motor_features.shape[1])
    time_encoder = TimeEmbedding(embedding_dim=16)
    fusion = IntermediateFusion(mask_missing=False)
    transformer = TransformerClassifier(input_dim=80)  # 32 (gait) + 32 (non-motor) + 16 (time)
    
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
    
    print(f"Starting Opals training with {len(train_features)} train samples, {len(val_features)} val samples")
    
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
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
        
        scheduler.step()
    
    print("Opals training complete.")
    
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
    
    print(f"\nFinal Opals Validation Results:")
    print(f"MSE: {final_mse:.4f}")
    print(f"MAE: {final_mae:.4f}")
    print(f"R²: {final_r2:.4f}")
    
    # Save model
    torch.save({
        'gait_encoder': gait_encoder.state_dict(),
        'non_motor_encoder': non_motor_encoder.state_dict(),
        'time_encoder': time_encoder.state_dict(),
        'fusion': fusion.state_dict(),
        'transformer': transformer.state_dict(),
        'scaler': scaler,
        'modality': modality_name,
        'metrics': {'mse': final_mse, 'mae': final_mae, 'r2': final_r2}
    }, 'opals_model.pth')
    
    print(f"Opals model saved as 'opals_model.pth'")

if __name__ == "__main__":
    main()
