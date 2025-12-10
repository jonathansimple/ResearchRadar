import os
import json
import numpy as np
import faiss
from openai import OpenAI

LLAMA_SERVER_URL = "http://localhost:11434/v1" # for ollama
EMBED_MODEL = "bge-m3:latest"

client = OpenAI(base_url=LLAMA_SERVER_URL, api_key="xxx")

INDEX_FILE = "store_router.index"
META_FILE = "store_metadata.json"

def list_files_in_folder(folder_path, recursive=False):
    """
    Finds all files in a given folder and returns a list of filenames.

    Args:
        folder_path (str): Path to the folder.
        recursive (bool): If True, includes files from subdirectories.

    Returns:
        list[str]: List of file names (with relative paths if recursive=True).
    """
    file_list = []

    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_list.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                file_list.append(full_path)

    return file_list


def build_router_index(rag_list):
    """
    Compute embeddings for each store name and build a FAISS index for routing.
    """
    print("🧠 Building FAISS router index...")
    store_embeddings = []
    for name in rag_list:
        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=name
        ).data[0].embedding
        store_embeddings.append(emb)

    # Convert to numpy array (float32 for FAISS)
    vectors = np.array(store_embeddings, dtype=np.float32)

    # Build FAISS index (L2 similarity)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Use inner product for cosine-like
    faiss.normalize_L2(vectors)
    index.add(vectors)

    # Save index + metadata
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"stores": rag_list}, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved router index to {INDEX_FILE} and metadata to {META_FILE}")

if __name__ == "__main__":
    rag_list = list_files_in_folder("./recent_arxiv_cs_AI")
    print("Found files:", rag_list)
    build_router_index(rag_list)

