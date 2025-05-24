import torch
import torch.nn as nn 
import transformers
from moe_mamba.model import MoEMamba
from transformers import AutoTokenizer

MAX_LENGTH = 512

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): The model to count parameters for.
        
    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Viper(nn.Module):
    def __init__(
        self, 
        config
    ):
        super(Viper, self).__init__()
        self.config = config

        self.model = MoEMamba(
            num_tokens=config['num_tokens'],
            dim = config['dim'],
            depth = config['depth'],
            num_experts = config['num_experts'],
            d_state = config['expert_dimension'],
            m_expand = config['expand'],
            causal = True,
            shared_qk = True,
            exact_window_size = True,
            dim_head = config["dim_head"]
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        if self.training:
            # predicted = outputs
            labels = input_ids.clone()
            loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            return outputs, loss
        else:
            return outputs
    
    @torch.no_grad()
    def generate(self, input_ids):
        pass

if __name__ == "__main__":
    import json
    cfg = {
        "dim": 512,
        "depth": 12,
        "num_experts": 4,
        "expert_dimension": 1024,
        "expand": 4,
        "num_tokens": 50729,
        "dim_head": 64
    }
    # num_tokens=50279,
    # dim=1024,
    # depth=1,
    # d_state=1024,
    # causal=True,
    # shared_qk=True,
    # exact_window_size=True,
    # dim_head=64,
    # m_expand=4,
    # num_experts=4,
    model = Viper(cfg)
    print("Total parameters:", count_parameters(model))
    tokenizer = AutoTokenizer.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/mamba_moelm/gpt_tokenizer/")
    x = tokenizer(
        "This is a test sentence.",
        return_tensors='pt'
    )
    input_ids = x['input_ids']
    # padding = torch.zeros((1, MAX_LENGTH - input_ids.shape[1]), dtype=torch.long)
    # input_ids = torch.cat([input_ids, padding], dim=1)
    print(input_ids)
    print(input_ids.shape)
    # loss = model.compute_loss(input_ids)
    # print(loss)
    model.train()
    model.to("cuda")
    input_ids = input_ids.to("cuda")
    outputs, loss = model(input_ids)
    print("Output shape:",outputs.shape)
    print("Loss:", loss.item())