import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import signal
import sys
from dataset import create_dataloader
from moe_lm.mamba_model import MambaModel
from tqdm import tqdm
import torch.nn as nn 
from transformers import AutoProcessor, AutoTokenizer
from model import ViperVL
from metric import perplexity
import logging



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def eval(model, val_loader, device):
    model.eval()
    total_perplexity = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            images = batch["pixel_values"].to(device)
            outputs = model(token_ids=input_ids, image=images)
            perplexity_value = perplexity(outputs, input_ids)
            total_perplexity += perplexity_value.item()
    return total_perplexity / len(val_loader)

def train(rank, world_size):
    print(f"[Rank {rank}] Starting training setup...", flush=True)
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    print(f"[Rank {rank}] Device set to {device}", flush=True)

    model_base = ViperVL(
        model_name="/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/mamba-1.5b",
        vision_ckpt_path="/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/vision_siglip",
        stage=1
    ).to(device)

    def handle_signal(sig, frame):
        print(f"[Rank {rank}] Received signal {sig}, shutting down gracefully.", flush=True)
        try:
            cleanup()
        except:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if rank == 0:
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            filename="logs/ddp_training.log",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    model = DDP(model_base, device_ids=[rank], find_unused_parameters=True)

    processor = AutoProcessor.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/vision_siglip")
    tokenizer = AutoTokenizer.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper_tokenier")

    dataloaders = create_dataloader(
        image_path="/home/mamba/ML_project/Testing/Huy/joint_vlm/dataset/cc_3m/images",
        json_path="/home/mamba/ML_project/Testing/Huy/joint_vlm/dataset/cc_3m/chat.json",
        tokenizer=tokenizer,
        processor=processor,
        batch_size=16,
        num_workers=16,
        ddp=True,
        rank=rank,
        world_size=world_size
    )

    train_loader = dataloaders["train_dataloader"]
    val_loader = dataloaders["val_dataloader"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    for epoch in range(10):
        model.train()
        total_loss = 0.0

        if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # ✅ Only rank 0 uses tqdm to avoid clutter
        if rank == 0:
            train_iterator = tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", unit="batch")
        else:
            train_iterator = train_loader
        for step, batch in enumerate(train_iterator):
            # if step == 10:
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

            # ✅ Update progress bar if rank 0
            if rank == 0 and isinstance(train_iterator, tqdm):
                train_iterator.set_postfix(loss=total_loss / (step + 1))

        scheduler.step()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}] Avg Train Loss: {avg_loss:.4f}", flush=True)
            avg_perplexity = eval(model, val_loader, device)
            print(f"[Epoch {epoch+1}] Validation Perplexity: {avg_perplexity:.4f}", flush=True)
            logging.info(f"[Epoch {epoch+1}] Avg Train Loss: {avg_loss:.4f}, Validation Perplexity: {avg_perplexity:.4f}")
            logging.info("======================================================")
            model.module.save_model(f"results/epoch_{epoch+1}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
