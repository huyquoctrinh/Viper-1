import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from moe_lm.mamba_model import MambaModel
from vision_encoder.siglip import VisionEncoder
from module import process_multimodal_input_ids
from module import ResidualFFNProjector
from constant.token_no import IMAGE_TOKEN
from transformers.image_utils import load_image
import os 
from tqdm import tqdm

class ViperVL(nn.Module):
    def __init__(self, model_name, vision_ckpt_path, stage = 1):
        super(ViperVL, self).__init__()
        self.lm_model = MambaModel.from_pretrained(model_name)
        # self.lm_model = self.model.cuda().half()
        self.vision_encoder = VisionEncoder(ckpt_path=vision_ckpt_path)
        self.projector = ResidualFFNProjector(
            input_dim=1152,
            output_dim=1152
        )

        if stage == 1:
            self.lm_model.eval()
            for param in self.lm_model.parameters():
                param.requires_grad = False
        else:
            self.lm_model.train()

        self.lm_model.resize_token_embeddings(1)

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.lm_model.save_pretrained(path + "/lm_model/")
        # torch.save(self.lm_model.state_dict(), f"{path}/lm_model.pth")
        torch.save(self.vision_encoder.state_dict(), f"{path}/vision_encoder.pth")
        torch.save(self.projector.state_dict(), f"{path}/projector.pth")

    def load_model(self, path, device="cuda"):
        # self.lm_model = MambaModel.from_pretrained(path + "/lm_model")
        self.lm_model.from_pretrained(checkpoint_name = path + "/lm_model/")
        self.vision_encoder.load_state_dict(torch.load(f"{path}/vision_encoder.pth"))
        self.projector.load_state_dict(torch.load(f"{path}/projector.pth"))
        # self.lm_model.eval()
        self.lm_model = self.lm_model.to(device)
        self.vision_encoder = self.vision_encoder.to(device)
        self.projector = self.projector.to(device)

    @torch.no_grad()
    def generate(
        self,
        input_ids, 
        image, 
        top_k = 50,
        max_length = 300,
        temperature = 1.
    ):
        print("Image shape:", image.shape)
        image_embeddings = self.vision_encoder(image)
        image_embeddings = self.projector(image_embeddings)

        # input_embeddings = self.lm_model.get_input_embeddings(token_ids)
        # generated_ids = input_ids.clone()
        generated = input_ids.clone()
        for _ in tqdm(range(max_length), desc = "Generating"):
            ids_embeds = self.lm_model.get_input_embeddings(generated)
            input_embeddings = process_multimodal_input_ids(
                batch_input_ids=generated,
                batch_input_ids_embedding=ids_embeds,
                image_embeddings=image_embeddings,
                image_tokens=IMAGE_TOKEN
            ).cuda().half()
            # print(input_embeddings.shape)
            outputs = self.lm_model(input_embeddings=input_embeddings)
            # print(outputs.shape)
            logits = outputs[:, -1,:]
            logits = logits/temperature
            probs = torch.softmax(logits, dim = 1)
            topk_logits, topk_indices = torch.topk(probs, top_k)
            sampled = torch.multinomial(topk_logits, num_samples=1)
            token_ids = topk_indices.gather(dim=1, index=sampled)
            generated = torch.cat([generated, token_ids], dim=-1)
        return generated

    def forward(self, image, token_ids):
        # Process image
        image_embeddings = self.vision_encoder(image)
        image_embeddings = self.projector(image_embeddings)
        image_embeddings = image_embeddings
        # Process token ids
        input_embeddings = self.lm_model.get_input_embeddings(token_ids)
        # Process multimodal input ids
        input_embeddings = process_multimodal_input_ids(
            batch_input_ids=token_ids,
            batch_input_ids_embedding=input_embeddings,
            image_embeddings=image_embeddings,
            image_tokens=IMAGE_TOKEN
        )
        # Forward pass through the language model
        input_embeddings = input_embeddings.cuda().half()
        out = self.lm_model(input_embeddings=input_embeddings)
        return out
    
# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    viper = ViperVL(
        model_name="/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/mamba-1.5b",
        vision_ckpt_path="/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/vision_siglip"
    )
    processor = AutoProcessor.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/vision_siglip")
    tokenizer = AutoTokenizer.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper_tokenier")

    image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
    image_inputs = processor(images=[image], return_tensors="pt").to("cuda")
    image_inputs = image_inputs["pixel_values"].to("cuda")
    text = "<image>Hello, how are you?"
    
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    token_ids = token_ids.to("cuda")
    
    viper = viper.cuda().half()
    output = viper(image_inputs, token_ids)
    print("Output shape:", output.shape)  # Should be [batch_size, sequence_length, hidden_size]
    

# model = MambaModel.from_pretrained(pretrained_model_name="/home/mamba/ML_project/Testing/Huy/joint_vlm/BlackMamba/pretrained/mamba-1.5b")
# model = model.cuda().half()
# inputs = torch.tensor([1, 2]).cuda().long().unsqueeze(0)
# input_embedding = model.get_input_embeddings(inputs).cuda().half()
# print("Input embedding shape:", input_embedding.shape)
# out = model(input_embeddings = input_embedding)
# print("Output shape:", out.shape)
