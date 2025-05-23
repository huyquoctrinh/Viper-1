import json

def load_json_data(json_path):
    """
    Load JSON data from the specified path.
    
    Args:
        json_path (str): Path to the JSON file.
        
    Returns:
        list: List of dictionaries containing the loaded data.
    """
    with open(json_path, 'r') as f:
        # data = json.load(f)
        data = [json.loads(line) for line in f if line.strip()]
    return data

data = load_json_data("/home/mamba/ML_project/Testing/Huy/joint_vlm/mamba_moelm/data/openweb_json/split/chunk_1.json")
# print(data.keys())
print(len(data))