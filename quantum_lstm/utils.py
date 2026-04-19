"""
Utility functions for Quantum LSTM
"""
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloaders(X, y, batch_size=64, val_split=0.2):
    """Create train/val dataloaders with proper scaling"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Train/Val split
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler

def benchmark_model(model, dataloader, device):
    """Benchmark model speed and memory"""
    model.eval()
    total_time = 0
    total_memory = 0
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(batch_x)
            end.record()
            
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)
            
            total_memory += torch.cuda.max_memory_allocated() / 1e9
    
    samples_per_sec = len(dataloader.dataset) / (total_time / 1000)
    return samples_per_sec, total_memory / len(dataloader)
