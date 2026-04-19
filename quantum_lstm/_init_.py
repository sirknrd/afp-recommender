
"""
Quantum LSTM Package - Production Fixed Version
"""
__version__ = "1.0.0"
__author__ = "sirknrd (fixed by AI)"

from .quantum_layer_fixed import OptimizedQuantumLSTM
from .utils import seed_everything, create_dataloaders

__all__ = ["OptimizedQuantumLSTM", "seed_everything", "create_dataloaders"]
