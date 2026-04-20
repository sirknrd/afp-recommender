#!/usr/bin/env python3
"""
Production Training Script
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.quantum_lstm_model_fixed import QuantumLSTMModel
import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = QuantumLSTMModel(num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Dummy data (reemplaza con tu dataset)
    train_data = torch.randn(1000, 20, 64)
    train_labels = torch.randn(1000, 1)
    
    writer = SummaryWriter('runs/quantum_lstm_fixed')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(train_data), args.batch_size):
            batch_x = train_data[i:i+args.batch_size].to(device)
            batch_y = train_labels[i:i+args.batch_size].to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = nn.MSELoss()(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / (len(train_data) // args.batch_size)
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), 'quantum_lstm_fixed.pth')
    print("✅ Model saved!")

if __name__ == "__main__":
    train()
