import argparse
import logging
import os

from tqdm import tqdm
import torch
import torch.nn as nn 
from transformers import AutoProcessor, AutoTokenizer
from dataset import create_dataloader

from model import ViperVL
from utils.metric import perplexity
from utils.misc import read_yaml_as_dict
from moe_lm.mamba_model import MambaModel


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval(
    model, 
    val_dataloader, 
    device
):
    model.eval()
    total_perplexity = 0.0
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", unit="batch"):
            # cnt+=1
            # if cnt == 3:
                # break
            input_ids = batch["input_ids"].to(device)
            images = batch["pixel_values"].to(device)
            outputs = model(token_ids=input_ids, image=images)
            perplexity_value = perplexity(outputs, input_ids)
            total_perplexity += perplexity_value.item()
    avg_perplexity = total_perplexity / len(val_dataloader)
    return avg_perplexity

def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=10
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        cnt = 0
        print(f"Trainable parameters: {count_trainable_parameters(model)}")
        for batch in progress_bar:
            # cnt += 1
            # if cnt == 3:
                # break
            input_ids = batch["input_ids"].to(device)
            images = batch["pixel_values"].to(device)
            optimizer.zero_grad()
            outputs = model(token_ids=input_ids, image=images)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            mean_curr_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=mean_curr_loss)

        scheduler.step()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        avg_perplexity = eval(model, val_dataloader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Perplexity: {avg_perplexity:.4f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Perplexity: {avg_perplexity:.4f}")
        print("======================================================")
        model.save_model(f"results/epoch_{epoch+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViperVL model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/huy.yml",
        help="Path to the configuration YAML file"
    )
    args = parser.parse_args()
    
    config = read_yaml_as_dict(args.config)
    
    # logging configuration
    logs_config = config.get("logging", {})
    logs_path = logs_config.get("filename", "logs/training.log")
    
    # model configuration
    model_config = config.get("model", {})
    model_name = model_config.get("model_name", "viper-vlm")
    vision_ckpt_path = model_config.get("ckpt_path", "pretrained/vision_siglip")
    tokenizer_path = model_config.get("tokenizer", "viper_tokenier")
    stage = model_config.get("stage", 1)
    
    # dataset configs
    dataset_config = config.get("dataset", {})
    image_path = dataset_config.get("image_path", "/home/mamba/ML_project/Testing/Huy/joint_vlm/dataset/cc_3m/images")
    json_path = dataset_config.get("json_path", "/home/mamba/ML_project/Testing/Huy/joint_vlm/dataset/cc_3m/chat.json")
    
    logs_dir = os.path.dirname(logs_path)
    os.makedirs(logs_dir, exist_ok=True) # make sure logs directory exists
    
    logging.basicConfig(
        filename=logs_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize model, processor, and tokenizer
    model = ViperVL(
        model_name=model_name,
        vision_ckpt_path=vision_ckpt_path,
        stage=stage
    ).cuda()

    processor = AutoProcessor.from_pretrained(vision_ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Create dataloaders
    dataloaders = create_dataloader(
        image_path=image_path,
        json_path=json_path,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=16,
        num_workers=16
    )

    train_dataloader = dataloaders["train_dataloader"]
    val_dataloader = dataloaders["val_dataloader"]

    # Set up optimizer, loss function, and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-6
    )

    # Count trainable parameters
    num_params = count_trainable_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    # Train the model
    train(model, train_dataloader, val_dataloader, optimizer, criterion, cosine_scheduler, device="cuda", num_epochs=10)

