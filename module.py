import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.fc(x))

class ResidualFFNProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualFFNProjector, self).__init__()
        self.projector = MLP(input_dim, output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        x = self.projector(x)
        residual = x
        x = self.ffn(x)
        x = x + residual
        return x

def process_multimodal_input_ids(
    batch_input_ids,
    batch_input_ids_embedding,
    image_embeddings,
    image_tokens
):
    # print(batch_input_ids_embedding.shape, image_embeddings.shape)
    
    multimodal_embeddings = []
    for i, batch in enumerate(batch_input_ids):
        multimodal_embedding = []
        for j, token_id in enumerate(batch):
            if token_id == image_tokens:
                # print("Image embedding shape:", image_embeddings[i].shape)
                multimodal_embedding.append(image_embeddings[i])
            else:
                # print("Token embedding shape:", batch_input_ids_embedding[i][j].shape)
                multimodal_embedding.append(batch_input_ids_embedding[i][j])
        multimodal_embeddings.append(torch.stack(multimodal_embedding))
    return torch.stack(multimodal_embeddings, dim=0)