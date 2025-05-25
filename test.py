from moe_lm.mamba_model import MambaModel
import torch
import torch.nn as nn

# class MoEMamba(nn.Module):
#     def __init__(self, model_name):
#         super(MoEMamba, self).__init__()
#         self.model = MambaModel.from_pretrained(model_name)
#         self.model = self.model.cuda().half()

    
    
#     def forward(self, inputs):
#         return self.model(input_embeddings = inputs)

#     @torch.no_grad()
#     def generate()

model = MambaModel.from_pretrained(pretrained_model_name="/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/mamba-1.5b")
model = model.cuda().half()
inputs = torch.tensor([1, 2]).cuda().long().unsqueeze(0)
input_embedding = model.get_input_embeddings(inputs).cuda().half()
print("Input embedding shape:", input_embedding.shape) #1 x 2 x 1152
out = model(input_embeddings = input_embedding)
print("Output shape:", out.shape)
model.save_pretrained("./test_pretrained")