import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)
    
                    # self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                #                               process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                #                               **factory_kwargs)
class RowParallelLinearFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, process_group=None):
        super(RowParallelLinearFFN, self).__init__()
        self.network = nn.Sequential(
            RowParallelLinear(input_dim, hidden_dim, bias=True, process_group=process_group, sequence_parallel=True),
            nn.ReLU(),
            RowParallelLinear(hidden_dim, output_dim, bias=True, process_group=process_group, sequence_parallel=True),
        )

    def forward(self, x):
        return self.network(x)


class SwitchMixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        expert_output_dim,
        num_experts,
        top_k=1,
    ):
        super(SwitchMixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: MLP to generate logits for expert selection
        self.router = nn.Linear(input_dim, num_experts)

        # Experts: a list of FeedForward networks
        self.experts = nn.ModuleList(
            [
                FeedForward(input_dim, hidden_dim, expert_output_dim)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)  # Flatten to [B*SEQLEN, dim]

        # Routing tokens to experts
        router_logits = self.router(x_flat)
        topk_logits, topk_indices = router_logits.topk(
            self.top_k, dim=1
        )
        topk_gates = F.softmax(
            topk_logits, dim=1
        )  # Normalizing the top-k logits

        # Initializing the output
        output_flat = torch.zeros(
            batch_size * seq_len,
            self.experts[0].network[-1].out_features,
            device=x.device,
        )

        # Distributing tokens to the experts and aggregating the results
        for i in range(self.top_k):
            expert_index = topk_indices[:, i]
            gate_value = topk_gates[:, i].unsqueeze(1)

            expert_output = torch.stack(
                [
                    self.experts[idx](x_flat[n])
                    for n, idx in enumerate(expert_index)
                ]
            )

            output_flat += gate_value * expert_output

        # Reshape the output to the original input shape [B, SEQLEN, expert_output_dim]
        output = output_flat.view(batch_size, seq_len, -1)
        return output
    

class RowParallelLinearSwitchMixtureOfExperts(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        expert_output_dim,
        num_experts,
        process_group=None,
        top_k=1,
    ):
        super(RowParallelLinearSwitchMixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: MLP to generate logits for expert selection
        # self.router = nn.Linear(input_dim, num_experts)
        self.router = RowParallelLinear(
            input_dim, num_experts, bias=True, process_group=process_group, sequence_parallel=True
        )

        # Experts: a list of FeedForward networks
        self.experts = nn.ModuleList(
            [
                RowParallelLinearFFN(input_dim, hidden_dim, expert_output_dim, process_group=process_group)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)  # Flatten to [B*SEQLEN, dim]

        # Routing tokens to experts
        router_logits = self.router(x_flat)
        topk_logits, topk_indices = router_logits.topk(
            self.top_k, dim=1
        )
        topk_gates = F.softmax(
            topk_logits, dim=1
        )  # Normalizing the top-k logits

        # Initializing the output
        output_flat = torch.zeros(
            batch_size * seq_len,
            self.experts[0].network[-1].out_features,
            device=x.device,
        )

        # Distributing tokens to the experts and aggregating the results
        for i in range(self.top_k):
            expert_index = topk_indices[:, i]
            gate_value = topk_gates[:, i].unsqueeze(1)

            expert_output = torch.stack(
                [
                    self.experts[idx](x_flat[n])
                    for n, idx in enumerate(expert_index)
                ]
            )

            output_flat += gate_value * expert_output

        # Reshape the output to the original input shape [B, SEQLEN, expert_output_dim]
        output = output_flat.view(batch_size, seq_len, -1)
        return output


if __name__ == "__main__":
# Example Usage
    batch_size = 32
    seq_len = 10
    input_dim = 512
    hidden_dim = 2048
    expert_output_dim = 1024
    num_experts = 4
    top_k = 1

    moe = RowParallelLinearSwitchMixtureOfExperts(
        input_dim, hidden_dim, expert_output_dim, num_experts, top_k
    )
    x = torch.rand(batch_size, seq_len, input_dim)  # Example input tensor
    output = moe(x)
    # print(output)
    print(x.shape)
    print(output.shape)



