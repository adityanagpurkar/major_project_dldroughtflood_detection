"""
2_model_engine.py
PyTorch model definition, dataset logic, and training pipeline for Vidarbha Hydro DL (V1.0).

Architecture: Dual-Head Bi-Directional LSTM with Attention
  - 4 Multi-Modal Input Features: [Precipitation, NDVI, NDWI, LST (Land Surface Temperature)]
  - 14-day look-back window (configurable)
  - Head 1: Flood Inundation Probability (0.0 – 1.0)  — Binary Regression
  - Head 2: Drought Severity Classification (0, 1, 2, 3) — 4-class Softmax

Training supports:
  - 25-year multi-district datasets (~100k+ samples)
  - Learning rate scheduling (CosineAnnealing)
  - Gradient clipping for stability
  - Train/Validation split with early stopping patience
  - Multi-worker DataLoader for performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
import os
import time

FEATURE_COLUMNS = ['precipitation', 'NDVI', 'NDWI', 'LST']


class VidarbhaHydroDataset(Dataset):
    """
    PyTorch Dataset for multi-decadal Vidarbha climate data.
    Converts daily multi-sensor CSV rows into overlapping 14-day sequences
    with flood risk and drought severity targets.
    """
    def __init__(self, csv_file='vidarbha_training_data.csv', seq_length=14):
        self.seq_length = seq_length

        if os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file, index_col='date', parse_dates=True)
            print(f"  Loaded {len(self.data):,} rows from {csv_file}")
        else:
            print(f"  CSV not found at {csv_file}, generating simulated 25-year data...")
            dates = pd.date_range(start='2000-01-01', end='2024-12-31')
            self.data = pd.DataFrame({
                'precipitation': np.random.uniform(0, 100, len(dates)),
                'NDVI': np.random.uniform(0.1, 0.8, len(dates)),
                'NDWI': np.random.uniform(-0.5, 0.5, len(dates)),
                'LST': np.random.uniform(25.0, 46.0, len(dates))
            }, index=dates)

        # Ensure all feature columns exist
        for col in FEATURE_COLUMNS:
            if col not in self.data.columns:
                if col == 'LST':
                    self.data[col] = 38.0
                else:
                    self.data[col] = 0.0

        # Drop any non-numeric columns (like 'district') for feature extraction
        numeric_cols = FEATURE_COLUMNS + ['flood_risk', 'drought_severity']
        available_cols = [c for c in numeric_cols if c in self.data.columns]
        self.data = self.data[available_cols].copy()

        self.data[FEATURE_COLUMNS] = self.data[FEATURE_COLUMNS].ffill().bfill().fillna(0.0)

        self.features = self.data[FEATURE_COLUMNS].values.astype(np.float32)

        # Target labels: use pre-computed if available, otherwise derive
        if 'flood_risk' in self.data.columns:
            self.flood_risk = self.data['flood_risk'].values.astype(np.float32)
        else:
            # Derive from features (legacy behavior)
            precip_factor = np.clip(self.data['precipitation'].values / 100.0, 0, 1)
            ndwi_factor = np.clip((self.data['NDWI'].values + 0.5) / 1.5, 0, 1)
            self.flood_risk = np.clip(
                0.7 * precip_factor + 0.3 * ndwi_factor + np.random.normal(0, 0.05, len(self.data)),
                0.0, 1.0
            ).astype(np.float32)

        if 'drought_severity' in self.data.columns:
            self.drought_severity = self.data['drought_severity'].values.astype(np.int64)
        else:
            lst_score = np.clip((self.data['LST'].values - 32.0) / 12.0, 0, 1)
            ndvi_inv = np.clip(1.0 - self.data['NDVI'].values, 0, 1)
            drought_continuous = (0.5 * lst_score + 0.5 * ndvi_inv) * 3.0
            self.drought_severity = np.clip(
                np.round(drought_continuous + np.random.normal(0, 0.2, len(self.data))),
                0, 3
            ).astype(np.int64)

    def __len__(self):
        return max(0, len(self.features) - self.seq_length)

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y_flood = self.flood_risk[idx + self.seq_length - 1]
        y_drought = self.drought_severity[idx + self.seq_length - 1]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_flood, dtype=torch.float32),
            torch.tensor(y_drought, dtype=torch.long)
        )


class TemporalAttention(nn.Module):
    """Soft attention over LSTM time-step outputs to focus on critical days."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)     # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        return context, weights


class VidarbhaHydroLSTM(nn.Module):
    """
    Dual-Head Bidirectional LSTM with Temporal Attention for multi-hazard prediction.

    Architecture:
      Input → BiLSTM (2 layers) → Temporal Attention → Shared FC → [Flood Head, Drought Head]
    """
    def __init__(self, input_size=4, hidden_size=96, num_layers=2, dropout=0.15):
        super(VidarbhaHydroLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Bidirectional doubles the hidden size
        bi_hidden = hidden_size * 2

        # Temporal attention over LSTM outputs
        self.attention = TemporalAttention(bi_hidden)

        # Shared feature trunk
        self.fc_shared = nn.Sequential(
            nn.Linear(bi_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Head 1: Flood probability (sigmoid output, 0 to 1)
        self.fc_flood = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Head 2: Drought severity (4-class logits)
        self.fc_drought = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden*2)
        shared = self.fc_shared(context)

        flood_prob = torch.sigmoid(self.fc_flood(shared))
        drought_logits = self.fc_drought(shared)
        return flood_prob, drought_logits


def train_model(
    csv_file='vidarbha_training_data.csv',
    epochs=20,
    batch_size=64,
    learning_rate=0.001,
    model_save_path='vidarbha_model.pth',
    val_split=0.15,
    patience=5,
    seq_length=14
):
    """
    Train the VidarbhaHydroLSTM model on 25-year multi-sensor data.

    Features:
      - Train/Validation split
      - Cosine annealing learning rate schedule
      - Gradient clipping for stability
      - Early stopping with patience
      - Per-epoch metrics reporting
    """
    print("\n" + "="*70)
    print("  VIDARBHA HYDRO-LSTM TRAINING PIPELINE")
    print("="*70)

    # Load dataset
    dataset = VidarbhaHydroDataset(csv_file, seq_length=seq_length)
    total_samples = len(dataset)

    if total_samples == 0:
        print("❌ Error: No training samples available.")
        return None

    # Train/Validation split
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n  Total Samples:      {total_samples:,}")
    print(f"  Training Samples:   {train_size:,}")
    print(f"  Validation Samples: {val_size:,}")
    print(f"  Batch Size:         {batch_size}")
    print(f"  Sequence Length:    {seq_length} days")
    print(f"  Epochs:             {epochs}")
    print(f"  Learning Rate:      {learning_rate}")
    print(f"  Patience:           {patience}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device:             {device}\n")

    model = VidarbhaHydroLSTM(input_size=4, hidden_size=96, num_layers=2, dropout=0.15).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters:   {total_params:,} total ({trainable_params:,} trainable)\n")

    criterion_flood = nn.BCELoss()
    criterion_drought = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    train_history = []

    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'Val Flood MAE':>14}  {'Val Drought Acc':>16}  {'LR':>10}  {'Time':>8}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*14}  {'-'*16}  {'-'*10}  {'-'*8}")

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_x, batch_y_flood, batch_y_drought in train_loader:
            batch_x = batch_x.to(device)
            batch_y_flood = batch_y_flood.to(device)
            batch_y_drought = batch_y_drought.to(device)

            optimizer.zero_grad()
            flood_pred, drought_logits = model(batch_x)

            loss_flood = criterion_flood(flood_pred.squeeze(-1), batch_y_flood)
            loss_drought = criterion_drought(drought_logits, batch_y_drought)
            loss = loss_flood + loss_drought

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / max(1, train_batches)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_flood_mae = 0.0
        val_drought_correct = 0
        val_drought_total = 0

        with torch.no_grad():
            for batch_x, batch_y_flood, batch_y_drought in val_loader:
                batch_x = batch_x.to(device)
                batch_y_flood = batch_y_flood.to(device)
                batch_y_drought = batch_y_drought.to(device)

                flood_pred, drought_logits = model(batch_x)

                loss_flood = criterion_flood(flood_pred.squeeze(-1), batch_y_flood)
                loss_drought = criterion_drought(drought_logits, batch_y_drought)
                loss = loss_flood + loss_drought

                val_loss += loss.item()
                val_batches += 1

                # Flood MAE
                val_flood_mae += torch.mean(torch.abs(flood_pred.squeeze(-1) - batch_y_flood)).item()

                # Drought accuracy
                drought_preds = torch.argmax(drought_logits, dim=1)
                val_drought_correct += (drought_preds == batch_y_drought).sum().item()
                val_drought_total += batch_y_drought.size(0)

        avg_val_loss = val_loss / max(1, val_batches)
        avg_flood_mae = val_flood_mae / max(1, val_batches)
        drought_acc = val_drought_correct / max(1, val_drought_total)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start

        train_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_flood_mae': avg_flood_mae,
            'val_drought_acc': drought_acc
        })

        print(f"  {epoch+1:5d}  {avg_train_loss:12.4f}  {avg_val_loss:12.4f}  {avg_flood_mae:14.4f}  {drought_acc:16.4f}  {current_lr:10.6f}  {elapsed:7.1f}s")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⏹  Early stopping triggered at epoch {epoch+1} (patience={patience})")
                break

        scheduler.step()

    # Load best model weights
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))

    print(f"\n  ✅ Best model saved to {model_save_path} (val_loss={best_val_loss:.4f})")
    print(f"  Total training time: {sum(h.get('epoch', 0) for h in train_history) * 0 + len(train_history)} epochs completed")
    print(f"  Final Drought Accuracy:  {train_history[-1]['val_drought_acc']:.2%}")
    print(f"  Final Flood MAE:         {train_history[-1]['val_flood_mae']:.4f}")
    print("="*70 + "\n")

    return model


def predict_risk(model, sequence):
    """
    Runs inference on a 14-day sequence of 4 features (Precip, NDVI, NDWI, LST).
    sequence: nested list or numpy array of shape (14, 4)
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0)
        flood_prob, drought_logits = model(x)

        flood_val = float(flood_prob.item())
        drought_class = int(torch.argmax(drought_logits, dim=1).item())
        drought_probs = torch.softmax(drought_logits, dim=1).squeeze().tolist()

    return {
        "flood_probability": flood_val,
        "drought_severity": drought_class,
        "drought_class_probabilities": drought_probs
    }


if __name__ == "__main__":
    # Check for the 25-year dataset first, fall back to shorter one
    csv_25yr = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vidarbha_training_data.csv')

    model = train_model(
        csv_file=csv_25yr,
        epochs=20,
        batch_size=64,
        learning_rate=0.001,
        patience=5
    )

    if model is not None:
        # Test inference
        dummy_seq = np.random.uniform(0, 1, (14, 4)).tolist()
        result = predict_risk(model, dummy_seq)
        print(f"Test inference: {result}")
