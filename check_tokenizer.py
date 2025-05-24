from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.save_pretrained("./gpt_tokenizer")

# tokens = "<|endoftext|>"
# print(tokenizer.encode(tokens))