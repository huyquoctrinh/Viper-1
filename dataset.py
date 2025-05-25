import json 
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoModel, AutoProcessor
from torch.utils.data.distributed import DistributedSampler
from transformers.image_utils import load_image
from PIL import Image


PADDING_INDEX = 0

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

class CCDataset(Dataset):
    def __init__(
        self,
        image_path,
        json_path,
        tokenizer,
        processor
    ):
        self.image_path = image_path
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.data = load_json(json_path)

    def __len__(self):
        return len(self.data)
    
    def format_prompt(self, conversation):
        # This function can be customized to format the prompt as needed
        # print("Conversation:", conversation)
        prompt = conversation[0]["value"] + "\n" + conversation[1]["value"] + "<|eos|>"
        return prompt.strip()
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(f"{self.image_path}/{item['image']}").convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt")["pixel_values"][0]
        
        prompt = self.format_prompt(item["conversations"])

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        )["input_ids"][0]
        # print("Input IDs:", input_ids)
        # print("image shape:", inputs.shape)
        
        return {
            "input_ids": input_ids,
            "pixel_values": inputs.squeeze(0)
        }

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    padded_input_ids = []
    max_length = max(len(item["input_ids"]) for item in batch)
    for item in batch:
        input_ids = item["input_ids"]
        padded_input_ids.append(
            torch.cat(
                [input_ids, torch.tensor([PADDING_INDEX] * (max_length - len(input_ids)), dtype=torch.long)],
            )
        )
    padded_input_ids = torch.stack(padded_input_ids)
    # print(padded_input_ids)
    return {
        "input_ids": padded_input_ids,
        "pixel_values": pixel_values
    }

def create_dataloader(
    image_path,
    json_path,
    tokenizer,
    processor,
    batch_size = 32,
    num_workers = 4,
    ddp = False,
    rank = 0,
    world_size = 1
):

    dataset = CCDataset(
        image_path=image_path,
        json_path=json_path,
        tokenizer=tokenizer,
        processor=processor
    )

    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [0.95, 0.05]
    )
    
    if ddp:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader
    }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper_tokenier")
    processor = AutoProcessor.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper-vlm/pretrained/vision_siglip")
    
    dataloader = create_dataloader(
        image_path="/home/mamba/ML_project/Testing/Huy/joint_vlm/dataset/cc_3m/images/",
        json_path="/home/mamba/ML_project/Testing/Huy/joint_vlm/dataset/cc_3m/chat.json",
        tokenizer=tokenizer,
        processor=processor,
        batch_size=2
    )
    
    for batch in dataloader["train_dataloader"]:
        print(batch["input_ids"].shape, batch["pixel_values"].shape)
        # print("Input IDs:", batch["input_ids"])
        break