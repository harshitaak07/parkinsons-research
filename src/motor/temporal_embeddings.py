import os
import warnings
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Metadata (from your motor metadata) ----
IDENTIFIER_COLS = ['VISNO', 'patno', 'subject_id']

AXIVITY_FEATURE_GROUPS = {
    "data_quality": ['numberofdays', 'validdays6hr', 'validdays12hr',
                     'nonweardetected', 'upsidedowndetected'],
    "time_percent": [],  # dynamically filled if needed
    "svm": [],           # dynamically filled if needed
    "step_bout": [],     
    "rotation": [],      
    "gait": [],          
    "variability": [],   
    "std": []            
}

OPALS_FEATURE_GROUPS = {
    'Walking Speed and Cadence': ['SP_U', 'CAD_U'],
    'Arm Swing Amplitude and Variability': ['RA_AMP_U', 'LA_AMP_U', 'RA_STD_U', 'LA_STD_U'],
    'Symmetry and Asymmetry Measures': ['SYM_U', 'ASYM_IND_U'],
    'Stride and Step Timing and Regularity': ['STR_T_U', 'STR_CV_U', 'STEP_REG_U', 'STEP_SYM_U'],
    'Movement Smoothness / Jerk Measures': ['JERK_T_U', 'R_JERK_U', 'L_JERK_U'],
    'Functional Mobility (TUG) Test Metrics': ['TUG1_DUR', 'TUG1_STEP_NUM',
                                               'TUG1_STRAIGHT_DUR', 'TUG1_TURNS_DUR',
                                               'TUG1_STEP_REG', 'TUG1_STEP_SYM']
}

# ---- Dataset & Collate ----
class SubjectSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sub_col: str, ses_col: str, feature_cols: List[str], max_seq_len: int=16):
        if sub_col not in df.columns or ses_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{sub_col}' and '{ses_col}' columns.")
        if not feature_cols:
            raise ValueError("feature_cols must be provided and non-empty.")
        self.sub_col = sub_col
        self.ses_col = ses_col
        self.feature_cols = feature_cols
        self.max_seq_len = max_seq_len
        self.subjects, self.sequences, self.orig_indices = [], [], []

        for sub, subdf in df.groupby(sub_col):
            subdf_sorted = subdf.sort_values(by=ses_col, key=pd.to_datetime, errors='ignore')
            feats = subdf_sorted[self.feature_cols].to_numpy(dtype=np.float32)
            idxs = subdf_sorted.index.to_numpy(dtype=np.int64)
            if feats.shape[0] == 0: 
                continue
            self.subjects.append(sub)
            self.sequences.append(feats)
            self.orig_indices.append(idxs)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        idxs = self.orig_indices[idx]
        seq_len = seq.shape[0]
        if seq_len > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
            idxs = idxs[-self.max_seq_len:]
            seq_len = self.max_seq_len
        return {'sub': self.subjects[idx], 'seq': seq, 'len': seq_len, 'idxs': idxs}

def collate_fn(batch):
    batch_size = len(batch)
    seq_lens = [b['len'] for b in batch]
    max_len = max(seq_lens)
    feat_dim = batch[0]['seq'].shape[1]
    seqs = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
    masks = np.zeros((batch_size, max_len), dtype=np.float32)
    subs, idxs_list = [], []
    for i, b in enumerate(batch):
        L = b['len']
        seqs[i, :L] = b['seq']
        masks[i, :L] = 1.0
        subs.append(b['sub'])
        idxs_list.append(b['idxs'])
    return {'sub': subs, 'seq': torch.from_numpy(seqs), 'mask': torch.from_numpy(masks),
            'len': torch.tensor(seq_lens, dtype=torch.long), 'idxs': idxs_list}

# ---- Encoders ----
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=1, bidirectional=False, dropout=0.1, proj_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(self.out_dim, proj_dim)

    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        return self.proj(out)

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)

def next_step_prediction_loss(embeddings, mask):
    pred = embeddings[:, :-1, :]
    target = embeddings[:, 1:, :]
    valid_mask = mask[:, 1:] * mask[:, :-1]
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    mse = (pred - target).pow(2).mean(dim=2)
    return (mse * valid_mask).sum() / valid_mask.sum()

# ---- Motor Temporal Embeddings Function ----
def generate_motor_temporal_embeddings(
    input_csv: str,
    output_timepoint_csv: str,
    output_subject_csv: str,
    sub_col: str='subject_id',
    ses_col: str='VISNO',
    max_seq_len: int=16,
    encoder_type: str='transformer',
    embedding_dim: int=128,
    training_epochs: int=30,
    batch_size: int=32,
    learning_rate: float=1e-3,
    device: str='cuda' if torch.cuda.is_available() else 'cpu'
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)

    # Collect motor features dynamically from metadata
    motor_features = []
    for group in AXIVITY_FEATURE_GROUPS.values():
        motor_features.extend(group)
    for group in OPALS_FEATURE_GROUPS.values():
        motor_features.extend(group)
    motor_features = list(set(motor_features))

    if len(motor_features) == 0:
        raise ValueError("No motor features found in metadata.")

    # Dataset & Dataloader
    dataset = SubjectSeqDataset(df, sub_col=sub_col, ses_col=ses_col, feature_cols=motor_features, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    input_dim = len(motor_features)
    model = LSTMEncoder(input_dim=input_dim, hidden_dim=embedding_dim, proj_dim=embedding_dim) \
        if encoder_type.lower()=='lstm' else TransformerEncoderModel(input_dim=input_dim, d_model=embedding_dim,
                                                                     nhead=min(8, max(1, embedding_dim // 32)),
                                                                     num_layers=2, dim_feedforward=max(embedding_dim*2, 256))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)

    # Training loop
    for epoch in range(training_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            seq = batch['seq'].to(device)
            mask = batch['mask'].to(device)
            optimizer.zero_grad()
            emb = model(seq, mask)
            loss = next_step_prediction_loss(emb, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        scheduler.step(avg_loss)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{training_epochs}] avg_loss: {avg_loss:.6f}")

    # Generate embeddings
    subj_rows, out_rows = [], []
    dl_eval = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in tqdm(dl_eval, desc="Generating embeddings"):
            seq = batch['seq'].to(device)
            mask = batch['mask'].to(device)
            emb = model(seq, mask).cpu().numpy()
            mask_np = mask.cpu().numpy()
            for i, sub in enumerate(batch['sub']):
                valid_len = int(mask_np[i].sum())
                subj_embs = emb[i, :valid_len, :]
                idxs = batch['idxs'][i]
                for j in range(valid_len):
                    row = {'sub': sub, 'orig_index': int(idxs[j])}
                    for d in range(emb.shape[2]):
                        row[f'emb_{d+1}'] = float(subj_embs[j, d])
                    out_rows.append(row)
                pooled = subj_embs.mean(axis=0)
                subj_row = {'sub': sub}
                for d in range(emb.shape[2]):
                    subj_row[f'sub_emb_{d+1}'] = float(pooled[d])
                subj_rows.append(subj_row)

    tp_emb_df = pd.DataFrame(out_rows).merge(df.reset_index().rename(columns={'index':'orig_index'}),
                                             on='orig_index', how='left') if out_rows else pd.DataFrame(columns=['sub','orig_index'])
    subj_emb_df = pd.DataFrame(subj_rows).drop_duplicates(subset=['sub']).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_timepoint_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_subject_csv) or '.', exist_ok=True)
    tp_emb_df.to_csv(output_timepoint_csv, index=False)
    subj_emb_df.to_csv(output_subject_csv, index=False)
    print(f"Saved timepoint embeddings -> {output_timepoint_csv} ({tp_emb_df.shape})")
    print(f"Saved subject embeddings -> {output_subject_csv} ({subj_emb_df.shape})")
    return tp_emb_df, subj_emb_df

if __name__ == "__main__":
    input_csv = "data/processed/motor/motor_features.csv"
    output_timepoint_csv = "embeddings/motor_timepoint_embeddings.csv"
    output_subject_csv = "embeddings/motor_subject_embeddings.csv"

    generate_motor_temporal_embeddings(
        input_csv=input_csv,
        output_timepoint_csv=output_timepoint_csv,
        output_subject_csv=output_subject_csv,
        encoder_type='transformer',  # or 'lstm'
        training_epochs=5,  # use small number for quick test
        batch_size=16
    )
