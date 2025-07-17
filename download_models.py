import os
from sentence_transformers import SentenceTransformer
from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Download and save SentenceTransformer model
print('Downloading SentenceTransformer model...')
sb_model_name = 'paraphrase-MiniLM-L6-v2'
sb_model_path = os.path.join(MODELS_DIR, sb_model_name)
model = SentenceTransformer(sb_model_name)
model.save(sb_model_path)
print(f'SentenceTransformer model saved to {sb_model_path}')

# Download and save HuggingFace summarization model and tokenizer
print('Downloading HuggingFace summarization model and tokenizer...')
hf_model_name = 'facebook/bart-large-cnn'
hf_model_path = os.path.join(MODELS_DIR, 'bart-large-cnn')
model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model.save_pretrained(hf_model_path)
tokenizer.save_pretrained(hf_model_path)
print(f'HuggingFace model and tokenizer saved to {hf_model_path}')

print('All models downloaded and saved locally.') 