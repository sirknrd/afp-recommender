import pytest
import torch
import torchquantum.functional as tqf
from quantum_lstm.quantum_layer_fixed import OptimizedQuantumLSTM
from models.quantum_lstm_model_fixed import QuantumLSTMModel

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_batch():
    return torch.randn(32, 20, 64)  # batch, seq, features

def test_quantum_layer_shape(sample_batch, device):
    model = OptimizedQuantumLSTM(n_qubits=8).to(device)
    output = model(sample_batch.to(device))
    assert output.shape == (32, 20, 128)

def test_model_forward(sample_batch, device):
    model = QuantumLSTMModel(input_size=64, num_classes=1).to(device)
    output = model(sample_batch.to(device))
    assert output.shape == (32, 1)

def test_gradients_flow(sample_batch, device):
    model = OptimizedQuantumLSTM(n_qubits=8).to(device)
    output = model(sample_batch.to(device))
    loss = output.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()

def test_batch_sizes(device):
    sizes = [1, 16, 64, 128]
    for bs in sizes:
        data = torch.randn(bs, 10, 64).to(device)
        model = OptimizedQuantumLSTM().to(device)
        output = model(data)
        assert output.shape == (bs, 10, 128)
