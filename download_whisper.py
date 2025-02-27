from huggingface_hub import hf_hub_download
import os

repo_id = 'openai/whisper-large-v3'

files = ['merges.txt', 'preprocessor_config.json', 'model.safetensors', 'config.json', 'vocab.json',
         'tokenizer_config.json', 'normalizer.json', 'special_tokens_map.json', 'added_tokens.json', 'tokenizer.json']

if not os.path.exists("./pretrained/whisper-large-v3/"):
    os.makedirs("./pretrained/whisper-large-v3/")

for file in files:
    hf_hub_download(repo_id, filename=file, local_dir="./pretrained/whisper-large-v3/")
