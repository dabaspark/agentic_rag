import os
from sentence_transformers import SentenceTransformer

# Initialize embedding model
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

print("Loading embedding model (this might take a while the first time)...")
embedding_model = SentenceTransformer(
    'dunzhang/stella_en_400M_v5',
    trust_remote_code=True,
    cache_folder=MODEL_CACHE_DIR
)

# device='cpu' 