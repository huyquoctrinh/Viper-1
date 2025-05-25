import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
import torch.nn as nn
# load the model and processor

class VisionEncoder(nn.Module):
    def __init__(
        self,
        ckpt_path = "/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/vision_siglip"
    ):
        super(VisionEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(ckpt_path, device_map="cuda")

    def forward(self, x):
        with torch.no_grad():
            # print("Vision shape:", x.shape)
            image_embeddings = self.model.get_image_features(x)
        return image_embeddings

# ckpt = 
# model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
# processor = AutoProcessor.from_pretrained(ckpt)

if __name__ == "__main__":
    model = VisionEncoder().to("cuda")
    
    processor = AutoProcessor.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/vision_siglip")
# load the image
    image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
    inputs = processor(images=[image], return_tensors="pt").to("cuda")
    print(inputs.keys())
    # run infernece
    input_pixel_values = inputs["pixel_values"].to("cuda:0")
    with torch.no_grad():
        image_embeddings = model(input_pixel_values)    

    print(image_embeddings.shape) # 1x1024
