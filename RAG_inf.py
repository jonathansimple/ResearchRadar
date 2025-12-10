import os
from typing import List
import json
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig
import torch
import re
import openai
import faiss
import numpy as np

# doc to pdf -> pdf to json -> move corpus.json to /reproduce
#python main.py --dataset patent_auto --llm_base_url http://localhost:11434/v1 --llm_name gpt-oss:20b 

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging

DATA_DIR = "recent_arxiv_cs_AI_json"
STORE_DIR = "vectorstores"
LLAMA_SERVER_URL = "http://140.96.28.189:11434/v1" # for ollama
EMBED_MODEL = "bge-m3:latest"
client = openai.OpenAI(base_url=LLAMA_SERVER_URL, api_key="xxx")

# --- BUILD VECTOR STORES ---
def build_vector_stores():
    os.makedirs(STORE_DIR, exist_ok=True)
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".json"):
            continue

        file_path = os.path.join(DATA_DIR, file)
        store_name = os.path.splitext(file)[0]
        print(f"📘 Processing {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Assume each JSON file has a list of dicts or text chunks
        # Try to extract text fields intelligently
        if isinstance(data, list):
            texts = [json.dumps(item) if isinstance(item, dict) else str(item) for item in data]
        elif isinstance(data, dict):
            texts = [json.dumps(data)]
        else:
            texts = [str(data)]

        # Create embeddings for all chunks
        embeddings = []
        for t in texts:
            emb = np.array(
                client.embeddings.create(
                    model=EMBED_MODEL,
                    input=t
                ).data[0].embedding,
                dtype=np.float32
            )
            embeddings.append(emb)

        X = np.vstack(embeddings)
        faiss.normalize_L2(X)

        # Build FAISS index
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)

        # Save both index and metadata
        faiss.write_index(index, os.path.join(STORE_DIR, f"{store_name}.faiss"))
        with open(os.path.join(STORE_DIR, f"{store_name}_texts.json"), "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {store_name}.faiss ({len(texts)} entries)")

if __name__ == "__main__":
    build_vector_stores()

# def route_query(query: str, index, store_names):
#     """
#     Choose the most relevant vector store given a user query.
#     """
#     emb = np.array(
#         client.embeddings.create(
#             model=EMBED_MODEL,
#             input=query
#         ).data[0].embedding,
#         dtype=np.float32
#     ).reshape(1, -1)

#     faiss.normalize_L2(emb)
#     D, I = index.search(emb, k=1)  # top-1 match
#     best_idx = int(I[0][0])
#     best_score = float(D[0][0])
#     best_store = store_names[best_idx]

#     print(f"➡️ Selected store: {best_store} (score={best_score:.4f})")
#     return best_store, best_score




# def doc_RAG(query, dataset_name):
#     #parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
#     #parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
#     #parser.add_argument('--llm_base_url', type=str, default='https://api.openai.com/v1', help='LLM base URL')
#     #parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
#     #parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name') #nvidia/NV-Embed-v2
#     #parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
#     #                    help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
#     #parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
#     #args = parser.parse_args()

#     #dataset_name = args.dataset
#     embedding_name = 'nvidia/NV-Embed-v2'
#     save_dir = 'outputs'
#     llm_base_url = 'https://api.openai.com/v1'
#     llm_name = 'gpt-4o-mini'
#     openie_mode = 'online'
#     if save_dir == 'outputs':
#         save_dir = save_dir + '/' + dataset_name
#     else:
#         save_dir = save_dir + '_' + dataset_name

#     # Single file corpus
#     #corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"    
#     #with open(corpus_path, "r", encoding="utf-8") as f:
#     #    raw = f.read()
#     # Clean invalid control characters
#     #raw_cleaned = re.sub(r'[\x00-\x1F\x7F]', '', raw)
#     #corpus = json.loads(raw_cleaned)
    
#     # Multi file corpus
#     corpus_dir = f"recent_arxiv_cs_AI_json"
#     corpus = []

#     # loop through all files in directory
#     for filename in os.listdir(corpus_dir):
#         if filename.endswith("_corpus.json"):
#             corpus_path = os.path.join(corpus_dir, filename)
#             with open(corpus_path, "r", encoding="utf-8") as f:
#                 raw = f.read()
#             # clean invalid control characters
#             raw_cleaned = re.sub(r'[\x00-\x1F\x7F]', '', raw)
#             data = json.loads(raw_cleaned)

#             # merge into main corpus list
#             corpus.extend(data)

#     print(f"Loaded {len(corpus)} paragraphs from all *_corpus.json files.")
    
    
#     docs = [f"{doc['text']}" for doc in corpus] # Text only dataset
    
#     all_queries = [query]



#     config = BaseConfig(
#         save_dir=save_dir,
#         llm_base_url=llm_base_url,
#         llm_name=llm_name,
#         dataset=dataset_name,
#         embedding_model_name=embedding_name,
#         embedding_base_url="http://140.96.28.189:11434/v1",
#         force_index_from_scratch=False,  # ignore previously stored index, set it to False if you want to use the previously stored index and embeddings
#         force_openie_from_scratch=False,
#         rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
#         retrieval_top_k=5,
#         linking_top_k=10,
#         max_qa_steps=5,
#         qa_top_k=5,
#         graph_type="facts_and_sim_passage_node_unidirectional",
#         embedding_batch_size=1,
#         max_new_tokens=None,
#         corpus_len=len(corpus),
#         openie_mode=openie_mode
#     )

#     logging.basicConfig(level=logging.INFO)

#     hipporag = HippoRAG(global_config=config)

#     hipporag.index(docs)

#     # Retrieval and QA - 5 outputs
#     # keys: question, docs, answer, 
#     answers, cot, generation_stats  = hipporag.rag_qa(queries=all_queries, gold_docs=None, gold_answers=None) 
    
#     return answers[0].answer
#     for i in range(len(answers)):
#         #print(answers[i])
#         print("Question:", answers[i].question)
#         print("Answer:", answers[i].answer)
#         print("CoT:", cot[i])
#         print("docs:", answers[i].docs)
#         print("generation_stats:", generation_stats[i])
#         print("---------------"*5)

# if __name__ == "__main__":
#     print(doc_RAG('智慧結構導向的代理模型協作技術的技術亮點是什麼？', '智慧結構導向的代理模型協作技術'))
