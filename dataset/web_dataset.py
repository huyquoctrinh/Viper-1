import torch
import json
import os 
from torch.utils.data import Dataset

def load_json_data(list_of_json_path):
    """
    Load JSON data from the specified path.
    
    Args:
        json_path (str): Path to the JSON file.
        
    Returns:
        list: List of dictionaries containing the loaded data.
    """
    data = []
    for json_path in list_of_json_path:
        print(f"Loading data from {list_of_json_path}")
        with open(json_path, 'r') as f:
            # data = json.load(f)
            data.extend([json.loads(line) for line in f if line.strip()])
    return data

class TextDatset(Dataset):
    def __init__(self, list_of_json_path, tokenizer):
        self.data = load_json_data(list_of_json_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        inputs_ids = self.tokenizer(
            text,
            return_tensors='pt'
        )
        return {'input_ids': inputs_ids['input_ids'].squeeze(0)}
    
def collate_fn(batch_data):
    input_ids = [item['input_ids'] for item in batch_data]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    return {'input_ids': input_ids}

def create_dataloader(list_of_json_path, tokenizer, batch_size=32, shuffle=True):
    list_of_json_path = list_of_json_path[:2]
    dataset = TextDatset(list_of_json_path, tokenizer)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.95, 0.05])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=16,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=16,
        shuffle=True,
        collate_fn=collate_fn
    )
    return {
        'train_loader': train_loader,
        'val_loader': val_loader
    }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    root = "/home/mamba/ML_project/Testing/Huy/joint_vlm/mamba_moelm/data/openweb_json/split"
    json_list = os.listdir(root)[:2]
    # print(json_list)
    json_list = [os.path.join(root, json) for json in json_list]
    tokenizer = AutoTokenizer.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/mamba_moelm/gpt_tokenizer")
    loader = create_dataloader(json_list, tokenizer, batch_size=2)
    for batch in loader['train_loader']:
        print(batch["input_ids"].shape)
        break