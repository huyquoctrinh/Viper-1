from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper_tokenier")
data = tokenizer("<image>Hello, how are you? <|eos|>", return_tensors="pt")
print(data)

print(len(tokenizer))
# tokenizer.add_tokens(["<image>"])
# tokenizer.save_pretrained("/home/mamba/ML_project/Testing/Huy/joint_vlm/viper")
