import os
import warnings
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def _select_feature_columns(df: pd.DataFrame, prefixes: Tuple[str, ...], exclude_cols: List[str]):
    cols = []
    for p in prefixes:
        cols.extend([c for c in df.columns if c.startswith(p)])
    if len(cols) == 0:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cols = [c for c in numeric_cols if c not in exclude_cols]
    return cols

class SubjectSeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sub_col: str='sub', ses_col: str='ses',
                 feature_cols: List[str]=None, max_seq_len: int=16, sort_ses_as_str: bool=False):
        if sub_col not in df.columns or ses_col not in df.columns:
            raise ValueError(f"Dataframe must contain '{sub_col}' and '{ses_col}' columns.")
        self.feature_cols = feature_cols or []
        if not self.feature_cols:
            raise ValueError("feature_cols must be provided and non-empty.")
        self.max_seq_len = int(max_seq_len)
        self.sub_col = sub_col
        self.ses_col = ses_col
        self.subjects = []
        self.sequences = []
        self.orig_indices = []
        grp = df.groupby(self.sub_col)
        for sub, subdf in grp:
            try:
                subdf_sorted = subdf.sort_values(by=self.ses_col, key=pd.to_datetime)
            except Exception:
                subdf_sorted = subdf.sort_values(by=self.ses_col)
            feats = subdf_sorted[self.feature_cols].to_numpy(dtype=np.float32)
            idxs = subdf_sorted.index.to_numpy(dtype=np.int64)
            if feats.shape[0] == 0:
                continue
            self.subjects.append(sub)
            self.sequences.append(feats)
            self.orig_indices.append(idxs)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        seq = self.sequences[i]
        idxs = self.orig_indices[i]
        seq_len = seq.shape[0]
        if seq_len > self.max_seq_len:
            seq = seq[-self.max_seq_len:, :]
            idxs = idxs[-self.max_seq_len:]
            seq_len = self.max_seq_len
        return {'sub': self.subjects[i], 'seq': seq, 'len': seq_len, 'idxs': idxs}

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

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=1, bidirectional=False, dropout=0.1, proj_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(self.out_dim, proj_dim)

    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        emb = self.proj(out)
        return emb

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return out

def next_step_prediction_loss(embeddings, mask):
    pred = embeddings[:, :-1, :]
    target = embeddings[:, 1:, :]
    valid_mask = mask[:, 1:] * mask[:, :-1]
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    mse = (pred - target).pow(2).mean(dim=2)
    loss = (mse * valid_mask).sum() / valid_mask.sum()
    return loss

def generate_imaging_temporal_embeddings(
    input_csv: str,
    output_timepoint_csv: str,
    output_subject_csv: str,
    id_cols: Tuple[str, str]=('sub', 'ses'),
    feature_prefixes: Tuple[str, ...]=('ae_', 'pca_'),
    exclude_cols: Tuple[str,...]=('sub','ses'),
    max_seq_len: int=16,
    encoder_type: str='transformer',
    embedding_dim: int=128,
    training_epochs: int=30,
    batch_size: int=32,
    learning_rate: float=1e-3,
    device: str='cuda' if torch.cuda.is_available() else 'cpu',
    supervised_target: Optional[str]=None,
    save_intermediate: bool=False
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    sub_col, ses_col = id_cols
    if sub_col not in df.columns or ses_col not in df.columns:
        raise ValueError(f"Input CSV must contain id columns: {id_cols}")
    feature_cols = _select_feature_columns(df, prefixes=feature_prefixes, exclude_cols=list(exclude_cols))
    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found with prefixes {feature_prefixes} and excluding {exclude_cols}")
    print(f"Using {len(feature_cols)} feature columns for temporal modeling.")
    dataset = SubjectSeqDataset(df, sub_col=sub_col, ses_col=ses_col, feature_cols=feature_cols, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    input_dim = len(feature_cols)
    if encoder_type.lower() == 'lstm':
        model = LSTMEncoder(input_dim=input_dim, hidden_dim=embedding_dim, proj_dim=embedding_dim)
    else:
        model = TransformerEncoderModel(input_dim=input_dim, d_model=embedding_dim,
                                        nhead=min(8, max(1, embedding_dim // 32)),
                                        num_layers=2, dim_feedforward=max(embedding_dim*2, 256))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    for epoch in range(training_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            seq = batch['seq'].to(device)
            mask = batch['mask'].to(device)
            optimizer.zero_grad()
            emb = model(seq, mask)
            if supervised_target is None:
                loss = next_step_prediction_loss(emb, mask)
            else:
                lengths = batch['len'].to(device)
                B, T, D = emb.shape
                last_idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, D)
                pooled = emb.gather(1, last_idx).squeeze(1)
                subs = batch['sub']
                y_list = []
                for s in subs:
                    srows = df[df[sub_col] == s]
                    if supervised_target in srows.columns:
                        try:
                            yval = srows[supervised_target].astype(float).iloc[-1]
                        except Exception:
                            yval = srows[supervised_target].iloc[-1]
                    else:
                        yval = 0.0
                    y_list.append(yval)
                y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32, device=device)
                pred = nn.Linear(pooled.shape[1], 1).to(device)
                y_pred = pred(pooled).squeeze(1)
                loss = nn.MSELoss()(y_pred, y_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)
        scheduler.step(avg_loss)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{training_epochs}] avg_loss: {avg_loss:.6f}")
    model.eval()
    subj_rows = []
    out_rows = []
    dl_eval = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in tqdm(dl_eval, desc="Generating embeddings"):
            seq = batch['seq'].to(device)
            mask = batch['mask'].to(device)
            emb = model(seq, mask)
            emb = emb.cpu().numpy()
            mask_np = mask.cpu().numpy()
            subs = batch['sub']
            idxs_list = batch['idxs']
            B, T, D = emb.shape
            for i in range(B):
                valid_len = int(mask_np[i].sum())
                subj_embs = emb[i, :valid_len, :]
                idxs = idxs_list[i]
                for j in range(valid_len):
                    row = {'sub': subs[i], 'orig_index': int(idxs[j])}
                    for d in range(D):
                        row[f'emb_{d+1}'] = float(subj_embs[j, d])
                    out_rows.append(row)
                pooled = subj_embs.mean(axis=0)
                subj_row = {'sub': subs[i]}
                for d in range(D):
                    subj_row[f'sub_emb_{d+1}'] = float(pooled[d])
                subj_rows.append(subj_row)
    if len(out_rows) == 0:
        warnings.warn("No embeddings produced (empty dataset).")
        tp_emb_df = pd.DataFrame(columns=['sub','orig_index'])
    else:
        tp_emb_df = pd.DataFrame(out_rows)
        tp_emb_df = tp_emb_df.merge(df.reset_index().rename(columns={'index':'orig_index'}), on='orig_index', how='left', suffixes=('_emb',''))
        emb_cols = [c for c in tp_emb_df.columns if c.startswith('emb_')]
        orig_cols = [c for c in df.columns]
        tp_emb_df = tp_emb_df[orig_cols + emb_cols]
    subj_emb_df = pd.DataFrame(subj_rows).drop_duplicates(subset=['sub']).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_timepoint_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(output_subject_csv) or '.', exist_ok=True)
    tp_emb_df.to_csv(output_timepoint_csv, index=False)
    subj_emb_df.to_csv(output_subject_csv, index=False)
    print(f"Saved timepoint embeddings -> {output_timepoint_csv} ({tp_emb_df.shape})")
    print(f"Saved subject embeddings -> {output_subject_csv} ({subj_emb_df.shape})")
    return tp_emb_df, subj_emb_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate imaging temporal embeddings")
    parser.add_argument('--input', type=str, default='results/fused_features_with_embeddings.csv')
    parser.add_argument('--out_timepoint', type=str, default='results/imaging_timepoint_embeddings.csv')
    parser.add_argument('--out_subject', type=str, default='results/imaging_subject_embeddings.csv')
    parser.add_argument('--max_len', type=int, default=16)
    parser.add_argument('--encoder', type=str, default='transformer')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    generate_imaging_temporal_embeddings(
        input_csv=args.input,
        output_timepoint_csv=args.out_timepoint,
        output_subject_csv=args.out_subject,
        max_seq_len=args.max_len,
        encoder_type=args.encoder,
        training_epochs=args.epochs,
        batch_size=args.batch_size
    )
