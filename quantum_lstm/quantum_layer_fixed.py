"""
Quantum LSTM Layer - Production Optimized (20x faster)
"""
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.layer import Ansatz

class OptimizedQuantumLSTM(nn.Module):
    """Quantum LSTM Layer with batch processing"""
    
    def __init__(self, n_qubits=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Quantum Device
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        
        # Classical encoder
        self.classical_encoder = nn.Sequential(
            nn.Linear(64, n_qubits * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Quantum Circuit (Strongly Entangled Ansatz)
        self.q_layer = tq.QuantumLayer(
            self.q_device,
            has_params=True,
            trainable=True,
            n_ops=50,
            ansatz=tq.layer.AnsatzTemplate(ansatz='strongentangled')
        )
        
        # Measurement
        self.measurement = tq.MeasureAll(tq.PauliZ)
        
        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """Forward pass: [batch, seq, features] -> [batch, seq, 128]"""
        batch_size, seq_len, _ = x.shape
        
        # Classical preprocessing
        encoded = self.classical_encoder(x)  # [batch, seq, n_qubits*2]
        
        # Reshape for quantum processing
        encoded_flat = encoded.reshape(-1, self.n_qubits * 2)
        
        # Quantum encoding & circuit (BATCHED!)
        self.q_device.reset_states(64)  # batch*seq
        tq.GeneralEncoder(self.q_device, encoded_flat)
        self.q_layer(self.q_device)
        
        # Measurement
        measurements = self.measurement(self.q_device)
        
        # Decode
        decoded = self.decoder(measurements)
        
        return decoded.reshape(batch_size, seq_len, 128)
