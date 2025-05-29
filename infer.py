from moe_lm.mamba_model import MambaModel
from tqdm import tqdm
import torch.nn as nn 
from transformers import AutoProcessor, AutoTokenizer
from model import ViperVL
import torch
from PIL import Image

def format_prompt(chat_prompt):
        # This function can be customized to format the prompt as needed
        # print("Conversation:", conversation)
        prompt = chat_prompt + "<|eos|>"
        return prompt.strip()

def process_data(
    image_path, 
    prompt,
    processor,
    tokenizer
):
    image = Image.open(image_path).convert("RGB")
    input_images = processor(images=[image], return_tensors="pt")["pixel_values"][0].squeeze(0)
    prompt = format_prompt(prompt)
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    )["input_ids"][0]

    return input_ids.unsqueeze(0), input_images.unsqueeze(0)

def infer(
    prompt, 
    image_path,
    model,
    tokenizer, 
    image_processor,
    temperature = 0.8,
    max_length = 128,
    top_k = 50,
    device = "cuda"
):

    input_ids, image = process_data(
        image_path,
        prompt, 
        image_processor,
        tokenizer
    )
    input_ids = input_ids.to(device)
    image = image.to(device)

    output_ids = model.generate(
        input_ids, 
        image, 
        top_k = top_k,
        max_length = max_length,
        temperature = temperature
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    image_path = "/data2/niz3873/viper/Viper-LM/000000000285.jpg"
    prompt = "#User: What is the animal inside the image ?"
    device = "cuda"
    processor = AutoProcessor.from_pretrained("/data2/niz3873/viper/pretrained/vision_encoder")
    tokenizer = AutoTokenizer.from_pretrained("/data2/niz3873/viper/Viper-LM/joint_vlm/viper")

    model = ViperVL(
        model_name="/data2/niz3873/viper/pretrained/lm_model",
        vision_ckpt_path="/data2/niz3873/viper/pretrained/vision_encoder",
        stage=2
    ).to(device)
    model.lm_model.eval()
    model.load_model("/data2/niz3873/viper/Viper-LM/results_s2_new/epoch_1")
    model.eval()

    answer = infer(
        prompt, 
        image_path,
        model,
        tokenizer = tokenizer, 
        image_processor= processor,
        temperature = 0.9,
        max_length = 64,
        top_k = 50
    )

    print(answer)
