# gen_embeddings.py
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re

import faiss
import pickle
import os

DB = "arxiv.db"                     # path to your sqlite DB
TABLE = "arxiv_index"
TEXT_COLS = ["title","categories"]     # columns to embed
EMB_TABLE = "arxiv_index"
BATCH = 128
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = "arxiv.faiss"      # output index file
IDMAP_PATH = "arxiv.pkl"           # mapping: faiss_id -> rowid


# ---- helpers ----
def to_bytes(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()

def sanitize_identifier(name: str) -> str:
    # basic safety: no double quotes or semicolons
    if '"' in name or ";" in name:
        raise ValueError(f"Unsafe identifier: {name!r}")
    # optional: allow only reasonable characters (adjust if you have unicode)
    # Here we allow anything but double-quote/semicolon for unicode names.
    return name

# ---- main ----
def build_embeddings():
    # sanitize table and columns
    sanitize_identifier(TABLE)
    for c in TEXT_COLS:
        sanitize_identifier(c)

    # prepare column list safely (quote identifiers with double quotes)
    cols_quoted = ", ".join([f'"{c}"' for c in TEXT_COLS])  # safe because sanitized above

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print("Model dim:", dim)

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # create embeddings table (rowid maps to source rowid)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {EMB_TABLE} (
            rowid INTEGER PRIMARY KEY,
            embedding BLOB
        );
    """)
    conn.commit()
    
    # try to add the new column (only if it doesn’t exist)
    try:
        cur.execute(f'ALTER TABLE {TABLE} ADD COLUMN embedding BLOB;')
        conn.commit()
        print(f"✅ Added column 'embedding' to {TABLE}")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("ℹ️ Column 'embedding' already exists.")
        else:
            raise

    # fetch data
    sql = f'SELECT rowid, {cols_quoted} FROM "{TABLE}"'
    cur.execute(sql)
    rows = cur.fetchall()

    texts = []
    rowids = []
    for rid, *cols in rows:
        parts = [("" if v is None else str(v)) for v in cols]
        text = " ||| ".join(parts)         # separator to preserve field boundaries
        texts.append(text)
        rowids.append(rid)

        # batch encode
        if len(texts) >= BATCH:
            embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            for r, emb in zip(rowids, embs):
                #cur.execute(f"INSERT OR REPLACE INTO {EMB_TABLE} (rowid, embedding) VALUES (?, ?)", (r, to_bytes(emb)))
                cur.execute(f"UPDATE {TABLE} SET embedding = ? WHERE rowid = ?", (to_bytes(emb), r))
            conn.commit()
            texts = []
            rowids = []

    # final partial batch
    if texts:
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        for r, emb in zip(rowids, embs):
            #cur.execute(f"INSERT OR REPLACE INTO {EMB_TABLE} (rowid, embedding) VALUES (?, ?)", (r, to_bytes(emb)))
            cur.execute(f"UPDATE {TABLE} SET embedding = ? WHERE rowid = ?", (to_bytes(emb), r))
        conn.commit()

    conn.close()
    print("Done: embeddings created and stored in table:", EMB_TABLE)

def from_bytes(b):
    return np.frombuffer(b, dtype=np.float32)

def rebuild_index(db_path=DB, emb_table=EMB_TABLE, index_path=INDEX_PATH, idmap_path=IDMAP_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT rowid, embedding FROM {emb_table} ORDER BY rowid")
    rows = cur.fetchall()
    if not rows:
        raise SystemExit("No embeddings found in table. Ensure embeddings were generated.")

    # stack embeddings into array (N, d)
    vecs = []
    id_map = []
    for rid, blob in rows:
        emb = from_bytes(blob)
        vecs.append(emb)
        id_map.append(int(rid))
    vecs = np.vstack(vecs).astype('float32')
    # optional: ensure normalized (if your pipeline expects cosine via IP)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    d = vecs.shape[1]
    print(f"Building FAISS IndexFlatIP with {vecs.shape[0]} vectors of dim {d}...")
    index = faiss.IndexFlatIP(d)   # inner product on normalized vectors => cosine
    index.add(vecs)
    # write index and idmap
    faiss.write_index(index, index_path)
    with open(idmap_path, "wb") as f:
        pickle.dump(id_map, f)
    print("Index written to", os.path.abspath(index_path))
    print("ID map written to", os.path.abspath(idmap_path))
    conn.close()

if __name__ == "__main__":
    build_embeddings()
    rebuild_index()


