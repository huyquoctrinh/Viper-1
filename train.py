from metric import perplexity
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from dataset.web_dataset import create_dataloader
import json
from transformers import AutoTokenizer
from model import Viper
import logging
import os

logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def count_parameters(model):
    """
    Count the number of trainable parameters in the
    model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def eval(
    model,
    val_loader,
    device
):
    model.eval()
    total_perplexity = 0
    cnt = 0
    for batch in tqdm(val_loader, desc="Validation"):
        cnt+=1
        if cnt == 2:
            break
        input_ids = batch['input_ids'].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            perplexity_value = perplexity(outputs, input_ids)
            total_perplexity += perplexity_value.item()
    avg_perplexity = total_perplexity / len(val_loader)
    return avg_perplexity

def train(
    model,
    train_loader,
    scheduler,
    val_loader,
    optimizer,
    num_epochs,
    device
):
    """
    Train the model on the training set and validate on the validation set.
    
    Args:
        model: The model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        optimizer: Optimizer for training.
        num_epochs: Number of epochs to train.
        device: Device to train on (e.g., 'cuda' or 'cpu').
        
    Returns:
        None
    """
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        cnt = 0
        for batch in progress_bar:
            cnt+=1
            if cnt ==2:
                break
            optimizer.zero_grad()
            model.to(device)
            input_ids = batch['input_ids'].to(device)
            outputs, loss = model(input_ids)
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.mean().item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        eval_perplexity = eval(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Perplexity: {eval_perplexity:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Perplexity: {eval_perplexity:.4f}")
        logging.info("====================================")
        # Save the model checkpoint
        if not os.path.exists(f"./results/{epoch+1}"):
            os.makedirs(f"./results/{epoch+1}")
        torch.save(model.module.state_dict(), f"./results/epoch_{epoch+1}/model.pth")
        with open(f"./results/epoch_{epoch+1}/config.json", "w") as f:
            json.dump(model.config, f)
        print(f"Model saved at epoch {epoch+1}")
if __name__ == "__main__":
    
    cfg = {
        "dim": 512,
        "depth": 12,
        "num_experts": 4,
        "expert_dimension": 512,
        "expand": 2,
        "num_tokens": 50727,
        "dim_head": 64
    }

    model = Viper(
        cfg
    )
    print("Trainable parameters:", count_parameters(model))
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.eos_token_id = tokenizer.eos_token_id 
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = "/home/mamba/ML_project/Testing/Huy/joint_vlm/mamba_moelm/data/openweb_json/split"
    json_list = os.listdir(root)
    json_list = [os.path.join(root, json) for json in json_list]
    tokenizer = AutoTokenizer.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/mamba_moelm/gpt_tokenizer")
    print("Loading data...")
    loader = create_dataloader(
        json_list, 
        tokenizer, 
        batch_size= 4
    )
    print("Data loaded.")
    train_loader = loader['train_loader']
    val_loader = loader['val_loader']
    num_epochs = 10

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5
    )

    train(
        model,
        train_loader,
        cosine_scheduler,
        val_loader,
        optimizer,
        num_epochs,
        device
    )