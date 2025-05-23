from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/mamba_moelm/gpt_tokenizer")
# tokenizer.save_pretrained("./gpt_tokenizer")

tokens = "<|endoftext|>"
print(tokenizer.encode(tokens))