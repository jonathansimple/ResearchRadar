import faiss, pickle, sqlite3, numpy as np
from sentence_transformers import SentenceTransformer
import re

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
index_path = "arxiv.faiss"
idmap_path = "arxiv.pkl"
DB = "arxiv.db"
TABLE = "arxiv_index"
TOPK = 50   # how many vectors to retrieve (not rows)

model = SentenceTransformer(MODEL)

index = faiss.read_index(index_path)
with open(idmap_path, "rb") as f:
    id_map = pickle.load(f)  # id_map[vector_id] => (rowid, field_name)

def semantic_query(query_text, topk_vectors=250, topk_rows=10, sql_filter=None):
    q_emb = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    D, I = index.search(q_emb, topk_vectors)  # I shape (1, k)
    ids = I[0].tolist()
    sims = D[0].tolist()

    # aggregate per rowid -> take max similarity among its vectors
    agg = {}
    for vec_id, sim in zip(ids, sims):
        (rowid, field) = id_map[vec_id]
        if sql_filter:
            # optional: skip rows not matching filter (we'll check SQL filter later)
            pass
        if rowid not in agg or sim > agg[rowid]['score']:
            agg[rowid] = {'score': sim, 'best_field': field, 'vec_id': vec_id}

    # get top rows
    top_rows = sorted(agg.items(), key=lambda kv: kv[1]['score'], reverse=True)[:topk_rows]
    rowids = [rid for rid, info in top_rows]

    # optionally apply SQL metadata filter here (e.g., year) by querying the DB for these rowids and discarding those that don't match
    placeholders = ",".join("?" for _ in rowids)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute(f'SELECT rowid, * FROM "{TABLE}" WHERE rowid IN ({placeholders})', rowids)
    rows = cur.fetchall()
    # reorder by rowids order
    rows_by_id = {r[0]: r for r in rows}
    results = []
    for rid, info in top_rows:
        results.append({'row': rows_by_id.get(rid), 'score': info['score'], 'best_field': info['best_field']})
    conn.close()
    return results
    

def extract_like_conditions(sql):
    """
    Returns a list of tuples: (column_name, search_text)
    """
    pattern = re.compile(r'"([^"]+)"\s+LIKE\s+["\']%([^%]+)%["\']', re.IGNORECASE)
    return pattern.findall(sql)


def semantic_like_rowids_no_filter(column, text, k=100):
    # Load FAISS index + ID map
    index = faiss.read_index(index_path)
    with open(idmap_path, "rb") as f:
        id_map = pickle.load(f)

    qvec = model.encode([text], normalize_embeddings=True)
    qvec = np.array(qvec).astype('float32')
    
    D, I = index.search(qvec, k)
    rowids = [id_map[i] for i in I[0] if i in id_map]
    # Remove duplicates while preserving order
    seen = set()
    unique_rowids = []
    for r in rowids:
        if r not in seen:
            seen.add(r)
            unique_rowids.append(r)
    return unique_rowids

def semantic_like_rowids(column, text, k=200, min_similarity=0.66, adaptive=False, index=index_path, idmap=idmap_path):
    # Load FAISS index + ID map
    index = faiss.read_index(index)
    with open(idmap, "rb") as f:
        id_map = pickle.load(f)
        
    qvec = model.encode([text], normalize_embeddings=True)
    qvec = np.array(qvec).astype('float32')
    
    D, I = index.search(qvec, k)

    # if no good results, adapt threshold
    if adaptive:
        # get top-10 average as dynamic high-confidence zone
        top_mean = np.mean(D[0][:10])
        # set cutoff slightly below that
        print("DEBUG:", type(min_similarity), min_similarity, type(top_mean), top_mean)

        cutoff = max(min_similarity, top_mean * 0.8)
    else:
        cutoff = min_similarity
        
    rowids = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx not in id_map:
            continue
        if dist >= cutoff:  # assuming inner product / cosine
            rowids.append(id_map[idx])

    # deduplicate
    seen = set()
    rowids = [r for r in rowids if not (r in seen or seen.add(r))]

    print(f"✅ {len(rowids)} retained for '{text}' (cutoff={cutoff:.3f})")
    return rowids


def semantic_like(column, text, k=100, index=index_path):
    """
    Perform semantic 'LIKE' by embedding `text` and retrieving similar rows.
    """
    # Load FAISS index + ID map
    index = faiss.read_index(index)
    with open(idmap_path, "rb") as f:
        id_map = pickle.load(f)

    qvec = model.encode([text], normalize_embeddings=True)
    qvec = np.array(qvec).astype('float32')

    # FAISS search
    D, I = index.search(qvec, k)
    rowids = [id_map[i] for i in I[0]]

    # Retrieve results from SQLite
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    placeholders = ",".join("?" * len(rowids))
    sql = f'SELECT rowid, "{column}" FROM "{TABLE}" WHERE rowid IN ({placeholders})'
    cur.execute(sql, rowids)
    results = cur.fetchall()
    conn.close()

    print(f"\n🔍 Semantic matches for '{text}' in column '{column}':")
    for r in results[:10]:
        print(" ", r)
    return results
    
def handle_semantic_likes(sql):
    like_conditions = extract_like_conditions(sql)
    if not like_conditions:
        print("No LIKE clauses detected.")
        return
    
    all_results = []
    for column, search_text in like_conditions:
        results = semantic_like(column, search_text, index, k=20)
        all_results.extend(results)
    return all_results


def apply_limit(sql: str, limit: int | None = None) -> str:
    """
    Append a LIMIT clause safely to a SQL query string — only if not already present.
    
    Args:
        sql: The original SQL query string.
        limit: Integer row limit (optional).
    Returns:
        Modified SQL string with a LIMIT clause if one wasn’t already there.
    """
    if not limit:
        return sql.strip()

    # Remove trailing semicolon
    sql = sql.strip().rstrip(';')

    # Check if a LIMIT clause already exists (case-insensitive)
    if re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
        # Already has LIMIT — don’t add another
        return sql + ';'

    # Otherwise, safely append
    return f"{sql} LIMIT {int(limit)};"


def replace_like_with_rowids(sql, column, term, rowids):
    """
    Replace the LIKE '%term%' clause for a column with rowid filtering.
    Works even if there are spaces, newlines, or different quote types.
    """
    id_list = ",".join(map(str, rowids))
    pattern = rf'("{re.escape(column)}"\s+LIKE\s+["\']%{re.escape(term)}%["\'])'
    replacement = f"rowid IN ({id_list})"
    new_sql, n = re.subn(pattern, replacement, sql, flags=re.IGNORECASE)
    
    if n == 0:
        print(f"⚠️ Warning: pattern not found for {column} LIKE '%{term}%'")
    else:
        print(f"✅ Replaced {n} occurrence(s) of {column} LIKE '%{term}%' to row ID")
    
    return new_sql

def rewrite_sql_with_semantic_likes(sql, index, idmap):
    like_conditions = extract_like_conditions(sql)
    rewritten_sql = sql

    for column, term in like_conditions:
        rowids = semantic_like_rowids(column, term, index=index, idmap=idmap)
        if not rowids:
            rewritten_sql = replace_like_with_rowids(
                rewritten_sql, column, term, [0]  # returns no rows
            )
        else:
            rewritten_sql = replace_like_with_rowids(
                rewritten_sql, column, term, rowids
            )
    return rewritten_sql


def convert_like_to_instr(sql: str) -> str:
    """
    Converts LIKE '%text%' clauses into instr() equivalents:
      col LIKE '%abc%' → instr(col, 'abc') > 0
    Works for all quote variants and handles multiple clauses.
    """
    pattern = re.compile(
        r'([`"\w\u4e00-\u9fff]+)\s+LIKE\s+(["\'`])%(.+?)%\2',
        re.IGNORECASE
    )
    return pattern.sub(r'instr(\1, "\3") > 0', sql)


def run_semantic_sql_old(sql, db, index, idmap):
    rewritten_sql = rewrite_sql_with_semantic_likes(sql, index, idmap)
    print("Rewritten sql:", rewritten_sql)
    rewritten_sql = apply_limit(rewritten_sql, 5)
    print("LIMITED SQL:", rewritten_sql)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(rewritten_sql)
    result = cur.fetchall()
    conn.close()
    return result

def run_semantic_sql(sql, db, index, idmap):
    rewritten_sql = rewrite_sql_with_semantic_likes(sql, index, idmap)
    #print("Rewritten SQL:", rewritten_sql)
    rewritten_sql = apply_limit(rewritten_sql, 50)
    #print("LIMITED SQL:", rewritten_sql)

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(rewritten_sql)

    # ✅ Extract both results and column names
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description] if cur.description else []

    # ✅ Fallback logic: if no results, try instr() version
    if False:# not rows and "LIKE" in rewritten_sql.upper():
        print("⚠️ No rows returned — retrying with instr() version...")
        instr_sql = convert_like_to_instr(rewritten_sql)
        print("Retry SQL:", instr_sql)
        cur.execute(instr_sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []

    conn.close()
    return {"columns": columns, "rows": rows}

#sql = """SELECT *
#FROM "契約測試資料20250923"
#WHERE "契約名稱" LIKE '%光學技術%';
#"""

#sql = """SELECT 
#  SUM(CASE WHEN "契約名稱" LIKE '%技轉案%' THEN 1 ELSE 0 END) AS "技轉案數量",
#  SUM(CASE WHEN "契約名稱" LIKE '%推廣案%' THEN 1 ELSE 0 END) AS "推廣案數量",
#  SUM(CASE WHEN "契約名稱" LIKE '%技轉案%' THEN "契約總金額台幣" ELSE 0 END) AS "技轉案總金額",
#  SUM(CASE WHEN "契約名稱" LIKE '%推廣案%' THEN "契約總金額台幣" ELSE 0 END) AS "推廣案總金額"
#FROM "契約測試資料20250923";"""

#rows = run_semantic_sql(sql)
#print(rows)
#print(len(rows))

