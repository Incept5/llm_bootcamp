from transformers import AutoTokenizer

# Load tokenizers for different models
tokenizer_chatgpt = AutoTokenizer.from_pretrained("gpt2")  # Using GPT-2 as a proxy for ChatGPT
tokenizer_opt = AutoTokenizer.from_pretrained("facebook/opt-350m")  # Using OPT model as an alternative

# Sample text to tokenize
text = "Tokenization can vary significantly between different models."

# Tokenize the text using both tokenizers
tokens_chatgpt = tokenizer_chatgpt.tokenize(text)
token_ids_chatgpt = tokenizer_chatgpt.convert_tokens_to_ids(tokens_chatgpt)

tokens_opt = tokenizer_opt.tokenize(text)
token_ids_opt = tokenizer_opt.convert_tokens_to_ids(tokens_opt)

# Print the tokens and their corresponding IDs for both tokenizers
print("ChatGPT-like Tokenizer (GPT-2):")
print("Tokens:", tokens_chatgpt)
print("Token IDs:", token_ids_chatgpt)

print("\nOPT Tokenizer:")
print("Tokens:", tokens_opt)
print("Token IDs:", token_ids_opt)