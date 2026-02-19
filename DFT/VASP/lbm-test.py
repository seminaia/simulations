import torch

def objective_function(U):
    """Mock objective function: Replace with actual DFT+U calculation."""
    # Example: Minimize a quadratic loss (mock bandgap error)
    # Optimal U is [4.5, 3.0] for this toy example
    loss = torch.sum((U - torch.tensor([4.5, 3.0]))**2 + torch.randn(1).abs() * 0.1)  # Simulated noise
    return loss
    