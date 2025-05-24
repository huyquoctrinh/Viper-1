import torch
from moe_mamba.model import MoEMamba

def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create a tensor of shape (1, 1024, 512)
x = torch.randint(0, 50279, (1, 6))

# Create a MoEMamba model
model = MoEMamba(
    num_tokens=50279,
    dim=1024,
    depth=1,
    d_state=1024,
    causal=True,
    shared_qk=True,
    exact_window_size=True,
    dim_head=64,
    m_expand=4,
    num_experts=4,
)
model.train()
print(model)
print("Trainable parameters:", count_parameters(model))

# Forward pass
out = model(x)

# Print the shape of the output tensor
print(out)
