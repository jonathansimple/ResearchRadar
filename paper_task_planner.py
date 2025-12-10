import torch
import os
import faiss
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from sql_query import sql_query
from semantic_search import semantic_query, handle_semantic_likes, run_semantic_sql
import sqlite3
from rapidfuzz import process, fuzz
import re
# from RAG_inf import doc_RAG
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from datetime import datetime


llm = ChatOllama(
    model="gpt-oss:20b",
    base_url="http://140.96.28.189:11434",
    num_ctx=8192,
    num_predict=8192,
    temperature=0.0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


# ---------- CONFIG ----------
API_KEY = "sk-no-key-required" # Fake key (llama-server doesn’t validate keys)

# Change this to your Ollama/LM Studio's host url or use localhost if running on the same machine
LLAMA_SERVER_URL = "http://140.96.28.189:11434/v1" # for ollama
MODEL_NAME = "gpt-oss:20b" # the model identifier for the model you are using
EMBED_MODEL = "bge-m3:latest"
INDEX_FILE = "store_router.index"
META_FILE = "store_metadata.json"
client = openai.OpenAI(base_url=LLAMA_SERVER_URL, api_key="xxx")
today_str = datetime.now().strftime("%Y-%m-%d") # '2025-10-22'


common_words = {}

def inference(query_str: str):
    conversations = [
        {"role": "system", "content": "You are a chatbot that integrates information retrieved from SQL into your answers."},
        {"role": "user", "content": query_str},
    ]
    
    llm_answer = ""
    
    conversations = [
        SystemMessage(content="You are a chatbot that integrates information retrieved from SQL into your answers. Include all the provided rows into a table, unless they are completely irrelevant. Only use column headers from the returned SQL list. You must not ignore or delete any row entries from the sql. Do not attempt to generate sql commands to reproduce the answer or explain your SQL reasoning."),
        HumanMessage(content=query_str)
    ]

    result = llm.invoke(conversations)

    return result

def check_ans(query_str):
    conversations = [
        {"role": "system", "content": "You are a RAG evaluation assistant. You are given a question and an answer. Determine if the question has been sufficiently answered. Respond simply with 'Yes' if the question is answered or 'No' if additional documents need to be retrieved to answer the question."},
        {"role": "user", "content": query_str},
    ]
    
    client = openai.OpenAI(base_url=LLAMA_SERVER_URL, api_key=API_KEY)

    # Stream the LLM output on the terminal
    stream = client.chat.completions.create(model=MODEL_NAME, messages=conversations, max_tokens=1024, temperature=0.0, stream=True)
    
    llm_answer = ""
    print("Response:")
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            llm_answer += chunk.choices[0].delta.content
    print("\n")
    return llm_answer

def load_model(model_name="Snowflake/Arctic-Text2SQL-R1-7B", device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()
    return tokenizer, model

def make_prompt(schema: str, question: str) -> str:
    prompt = f"""### Schema
{schema}

### Question
{question} 

### SQL:"""
    return prompt

def inference_sql(tokenizer, model, prompt: str, max_new_tokens=1024, device="auto"):
    conversations = [
        SystemMessage(content=f"You are a arxiv paper aggregator text-to-sql agent for a SQLite databases. Generate a valid SQL command based on the user input and provided schema. The user may ask to filter papers by domain or keyword, try to include both the full name and abbreviation when indexing the database. The years in the database are in 西元, so make sure to convert the years to Gregorian first if the given year is in Taiwan/ROC format. Today's date is {today_str}. Do not leave comments in the SQL."),
        HumanMessage(content=prompt)
    ]

    result = llm.invoke(conversations)

    return result.content
    
def inference_check_mode(prompt: str):
    conversations = [
        SystemMessage(content="You are an agent manager. There are two agents available: a text-to-sql agent, and a traditional, document based RAG agent. The SQL contains lists of arxiv papers alongside relevant information such authors, publication date, paper categories. The documents are the individual papers published on arxiv. Determine if the user query should be served by the text-to-sql agent or the RAG agent or first the text-to-sql agent then the RAG agent for additional context on a particular project. Respond with 'SQL' for text-to-sql agent or 'RAG' for RAG agent or 'Both' for both."),
        HumanMessage(content=prompt)
    ]

    result = llm.invoke(conversations)

    return result.content

def rephrase_question(prompt: str):
    conversations = [
        SystemMessage(content="You are an intermediary agent. A prior agent has provided you with relevant SQL information to answer the user's query. Use the contents of the SQL to rephrase the query so that the RAG agent can better understand and respond to the question."),
        HumanMessage(content=prompt)
    ]

    result = llm.invoke(conversations)

    return result.content

def get_columns(db_path, table):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    cols = [row[1] for row in cur.fetchall()]
    conn.close()
    return cols


def correct_identifiers(sql, real_cols, threshold=75):
    # Pattern: match double-quoted substrings, but capture context before them
    # This lets us check whether it's preceded by AS or a literal pattern (%)
    pattern = re.compile(r'(?<!%)"([^"]+)"(?!%)', re.IGNORECASE)

    def replace_identifier(match):
        ident = match.group(1)
        before = sql[:match.start()]  # context before the quote

        # Skip if preceded by AS (case-insensitive)
        if re.search(r'\bAS\s*$', before, re.IGNORECASE):
            print(f"⏭️ Skipping alias: \"{ident}\"")
            return f'"{ident}"'

        # Skip if looks like a string literal pattern (e.g., "%光學%")
        if ident.startswith('%') and ident.endswith('%'):
            print(f"⏭️ Skipping string literal: \"{ident}\"")
            return f'"{ident}"'

        # Perform fuzzy matching
        col_variants = [ident]
        for keyword in common_words:
            print("keyword:", keyword)
            new_col = ident.replace(keyword, common_words[keyword])
            print("replaced string:", new_col)
            if new_col != keyword:
                col_variants.append(new_col)
                print("new col added")
        
        best_match = 0
        score = 0
        for col in col_variants:
            match_1, score_1, _ = process.extractOne(col, real_cols, scorer=fuzz.partial_ratio)
            print("matching col", col)
            print("match_1, score_1", match_1, score_1)
            if score_1 > score:
                score = score_1
                best_match = match_1
            
        #best_match, score, _ = process.extractOne(ident, real_cols, scorer=fuzz.partial_ratio)
        if score >= threshold:
            print(f"✅ Replacing '{ident}' → '{best_match}' (score={score})")
            return f'"{best_match}"'
        else:
            print(f"⚠️ Keeping '{ident}' (best={best_match}, score={score})")
            return f'"{ident}"'

    # Apply substitution
    return pattern.sub(replace_identifier, sql)


def extract_like_conditions(sql):
    """
    Returns a list of tuples: (column_name, search_text)
    """
    pattern = re.compile(r'"([^"]+)"\s+LIKE\s+["\']%([^%]+)%["\']', re.IGNORECASE)
    return pattern.findall(sql)

# def split_sql_commands(sql_text: str):
#     """
#     Split multi-command SQL text into individual statements.
#     Removes comments and blank lines safely.
#     """
#     # Remove comments (lines starting with --)
#     cleaned = re.sub(r'--.*', '', sql_text)
#     # Split on semicolons
#     parts = [p.strip() for p in cleaned.split(';') if p.strip()]
#     return parts


def split_sql_commands(sql_text: str):
    """
    Split SQL text into individual commands, ignoring semicolons inside strings.
    Removes '--' comments and blank lines.
    """
    # Remove single-line comments
    cleaned = re.sub(r'--.*', '', sql_text)
    commands = []
    current = []
    in_single_quote = False
    in_double_quote = False

    for char in cleaned:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        if char == ';' and not in_single_quote and not in_double_quote:
            cmd = ''.join(current).strip()
            if cmd:
                commands.append(cmd)
            current = []
        else:
            current.append(char)

    # Add the last command (no trailing semicolon)
    final_cmd = ''.join(current).strip()
    if final_cmd:
        commands.append(final_cmd)

    return commands


def sql_cleanup(sql: str) -> str:
    """
    Cleans and normalizes SQL generated by LLMs:
      - Removes parentheses and contents immediately after FROM clauses
      - Handles cases like: FROM (SELECT ...), FROM table(alias), FROM "table(alias)"
      - Converts equality comparisons ("col" = 'val') into LIKE patterns ("col" LIKE '%val%')
      - Works with any quote style and Unicode identifiers
    """

    # --- FROM clause cleanup ---
    def clean_from_target(target: str) -> str:
        target = re.sub(r'\([^)]*\)', '', target)  # remove parentheses
        target = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', target)  # clean spaces in quoted names
        return target.strip()

    def from_replacer(match):
        raw_target = match.group(1)
        return f"FROM {clean_from_target(raw_target)} "

    from_pattern = re.compile(r'(?i)FROM\s+([^;\n]+)')
    sql = from_pattern.sub(from_replacer, sql)

    # --- Equality-to-LIKE conversion ---
    # Breaks on chinese half vs full width chars, use instr
    def like_replacer(match):
        col = match.group(1).strip()
        val = match.group(2).strip().strip("'\"`")
        return f"{col} LIKE '%{val}%'"

    #like_pattern = re.compile(r'([`"\w\u4e00-\u9fff]+)\s*=\s*(["\'`][^"\']+["\'`])')
    #sql = like_pattern.sub(like_replacer, sql)

    # Pattern explanation:
    #  (?!負責人姓名)   → negative lookahead to skip that column
    #  ([`"'\w\u4e00-\u9fff]+) → match any column name
    #  \s*=\s*          → equals sign
    #  (["'`][^"'`]+["'`]) → quoted value
    like_pattern = re.compile(
        r'(?<![^\s])(?!(負責人姓名))([`"\w\u4e00-\u9fff]+)\s*=\s*(["\'`][^"\']+["\'`])'
    )

    # We’ll apply manually: the pattern’s groups differ because of the (?!負責人姓名)
    def safe_replacer(match):
        col = match.group(2).strip()
        val = match.group(3).strip().strip("'\"`")
        return f"{col} LIKE '%{val}%'"

    sql = like_pattern.sub(safe_replacer, sql)
    return sql




def convert_like_to_instr(sql: str) -> str:
    """
    Converts LIKE clauses into Unicode-safe instr() equivalents.
      "契約類別" LIKE '%技轉%'  →  instr("契約類別", '技轉') > 0
    Works for single, double, or backtick quotes.
    """

    def replacer(match):
        col = match.group(1).strip()
        val = match.group(3).strip()
        # strip % wildcards, since instr() does substring search inherently
        val = val.strip("%")
        return f"instr({col}, '{val}') > 0"

    # Pattern explanation:
    #   ([`"\w\u4e00-\u9fff]+)   → column name, supports Chinese and quoted identifiers
    #   \s+LIKE\s+               → LIKE operator
    #   (["'`])([^"'`]+)\2       → quoted string pattern ('%技轉%' etc.)
    pattern = re.compile(r'([`"\w\u4e00-\u9fff]+)\s+LIKE\s+(["\'`])([^"\']+)\2', re.IGNORECASE)

    return pattern.sub(replacer, sql)

def convert_roc_years_old(sql: str) -> str:
    """
    Converts ROC years (e.g., 114 -> 2025) inside SQL strings.
    Also adjusts substr(…, 1, 3) -> substr(…, 1, 4) when expanding to 4-digit years.
    """

    def repl(match):
        num = match.group(2)
        try:
            year = int(num)
            if 1 <= year < 200:
                return f"{match.group(1)}{year + 1911}{match.group(3)}"
        except ValueError:
            pass
        return match.group(0)

    # Step 1: Convert 3-digit ROC years in comparison contexts
    sql = re.sub(r"([=<>]\s*[\n\r\s]*['\"]?)(\d{3})(['\"]?)", repl, sql)

    # Step 2: Adjust substring length if present (1, 3) → (1, 4)
    # Only when the pattern is looking for 3 leading digits
    sql = re.sub(r"substr\(([^,]+),\s*1\s*,\s*3\)", r"substr(\1, 1, 4)", sql)

    return sql


def convert_roc_years(sql: str) -> str:
    """
    Converts ROC years (e.g., 114 → 2025) only inside SQL literals and comparisons,
    skipping table names or quoted identifiers like "109-114歷年契約查詢20250919".
    """

    def repl_num(match):
        num = match.group(2)
        try:
            year = int(num)
            if 1 <= year < 200:
                return f"{match.group(1)}{year + 1911}{match.group(3)}"
        except ValueError:
            pass
        return match.group(0)

    def repl_str(match):
        num = match.group(1)
        try:
            year = int(num)
            if 1 <= year < 200:
                return str(year + 1911)
        except ValueError:
            pass
        return match.group(0)

    # Convert numeric comparisons (= 114, > 113, etc.)
    sql = re.sub(r"([=<>]\s*[\n\r\s]*['\"]?)(\d{3})(['\"]?)", repl_num, sql)

    # Convert string literals like '114', but skip those in FROM/JOIN lines
    def string_replacer(match):
        before = match.group(1)
        num = match.group(2)
        after = match.group(3)

        # Skip cases where the preceding text suggests identifiers
        if re.search(r"\bFROM\b|\bJOIN\b", before, flags=re.IGNORECASE):
            return match.group(0)

        try:
            year = int(num)
            if 1 <= year < 200:
                return f"{before}{year + 1911}{after}"
        except ValueError:
            pass
        return match.group(0)

    sql = re.sub(
        r"([^A-Za-z0-9_])(['\"])(\d{3})(['\"])",
        string_replacer,
        sql
    )

    # Fix substr(…, 1, 3) → substr(…, 1, 4)
    sql = re.sub(r"substr\(([^,]+),\s*1\s*,\s*3\)", r"substr(\1, 1, 4)", sql)

    return sql


def quote_special_identifiers(sql: str) -> str:
    """
    Ensures tables or columns with digits, Chinese chars, or symbols are quoted.
    Example: FROM 114年科專計畫清單_1140915 → FROM "114年科專計畫清單_1140915"
    """
    pattern = re.compile(r'\bFROM\s+([A-Za-z0-9\u4e00-\u9fff_]+)')
    def repl(match):
        name = match.group(1)
        if not name.startswith('"'):
            return f'FROM "{name}"'
        return match.group(0)
    return pattern.sub(repl, sql)








def load_router():
    """
    Load FAISS router and metadata.
    """
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        raise FileNotFoundError("Please build the router index first with build_router_index()")

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["stores"]

def route_query(query: str, index, store_names):
    """
    Choose the most relevant vector store given a user query.
    """
    emb = np.array(
        client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        ).data[0].embedding,
        dtype=np.float32
    ).reshape(1, -1)

    faiss.normalize_L2(emb)
    D, I = index.search(emb, k=1)  # top-1 match
    best_idx = int(I[0][0])
    best_score = float(D[0][0])
    best_store = store_names[best_idx]

    print(f"➡️ Selected store: {best_store} (score={best_score:.4f})")
    return best_store, best_score

rag_index, store_names = load_router()


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
                file_list.append(file)
    else:
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path):
                file_list.append(full_path)

    return file_list
    


# Load all vector stores
def load_vector_stores(store_dir="vectorstores"):
    stores, store_names = [], []
    for file in os.listdir(store_dir):
        if file.endswith(".faiss"):
            path = os.path.join(store_dir, file)
            store_name = os.path.splitext(file)[0]
            stores.append(faiss.read_index(path))
            store_names.append(store_name)
    return stores, store_names

# Example usage
RAG_stores, RAG_store_names = load_vector_stores()

# Build routing index (mean embedding of each store)
def build_router_index(stores, store_names):
    dim = stores[0].d
    router_index = faiss.IndexFlatIP(dim)
    store_embs = []

    for store in stores:
        xb = store.reconstruct_n(0, store.ntotal)
        mean_vec = xb.mean(axis=0, keepdims=True)
        store_embs.append(mean_vec)

    X = np.vstack(store_embs).astype(np.float32)
    faiss.normalize_L2(X)
    router_index.add(X)
    return router_index

router_index = build_router_index(RAG_stores, RAG_store_names)



def inf_end_to_end(user_input="Test"):
    questions = [user_input
    #"list out the papers related to RAG",
    #"list out the papers published in the last 2 months",
    ]


    #tokenizer, model = load_model()
    tokenizer = None
    model = None
    table_list = {"arxiv_index": 0}
    schema_list = ["Table: arxiv_index(id, title, authors, published, categories, pdf_url)"]
    db_list = ["arxiv.db"]
    index_list = ["arxiv.faiss"]
    idmap_list = ["arxiv.pkl"]
    rag_list = list_files_in_folder("./recent_arxiv_cs_AI")
    real_cols_list = [get_columns(db_list[table_list[table]], table) for table in table_list]
    print(real_cols_list)

    for question in questions:
        print("Query:",question)
        
        mode = inference_check_mode(question)
        print(f"-------------- USE {mode} -----------------")
        
        if "RAG" in mode:
            # store, score = route_query(question, rag_index, store_names)
            best_store, score = route_query(question, router_index, RAG_store_names)
            chosen_idx = RAG_store_names.index(best_store)
            index = RAG_stores[chosen_idx]

            # Load corresponding texts
            with open(f"vectorstores/{best_store}_texts.json", "r", encoding="utf-8") as f:
                texts = json.load(f)

            # Search inside selected store
            emb = np.array(
                client.embeddings.create(model=EMBED_MODEL, input="find recent RAG papers").data[0].embedding,
                dtype=np.float32
            ).reshape(1, -1)
            faiss.normalize_L2(emb)
            D, I = index.search(emb, k=50)
            context_string = ""
            for rank, idx in enumerate(I[0]):
                context_string = context_string + texts[idx] + "\n"

            print("context: ", context_string)
            print("-------------------")

    
            conversations = [
                SystemMessage(content="You are a chatbot that integrates information retrieved from RAG into your answers. Use only relevant information from the retrieved chunks."),
                HumanMessage(content=question + "\n Contexts:\n" + context_string)
            ]

            output = llm.invoke(conversations).content

            # output = doc_RAG(question, store)
            #print(output)
            #continue
            return output
        
            
        
        schema_queue = []
        row_list = []
        used_table_list = []
        schema_queue.append(schema_list[0])
        
        print("USING SCHEMA:", schema_queue)
        
        for schema in schema_queue:
            prompt = make_prompt(schema, question)
        
            print("----> text to sql prompt:", prompt)
        
            sql = inference_sql(tokenizer, model, prompt)
        
            #parse sql
            if r"```sql" in sql:
                sql = sql.split(r"```sql")[-1].split(r"```")[0][:-1]
            sql = sql_cleanup(sql)
            sql = convert_roc_years(sql)
            sql = quote_special_identifiers(sql)
            sql = convert_like_to_instr(sql)
        
            sql_list = split_sql_commands(sql)
            print("Parsed SQL commands:", len(sql_list))
        
            
            for sql in sql_list:
                # Determine which table is being queried
                real_cols = []
                for table in table_list:
                    if table in sql:
                        used_table_list.append(table)
                        db = db_list[table_list[table]]
                        index = index_list[table_list[table]]
                        idmap = idmap_list[table_list[table]]
                        real_cols = real_cols_list[table_list[table]]

        
                sql = correct_identifiers(sql, real_cols) #may delete subsequent lines if there are comments and no line breaks after each command
                print("---------------- FINAL SQL -----------------")
                print(sql)
                print("---------------- FINAL SQL -----------------")


                rows = run_semantic_sql(sql, db, index, idmap)
                #if not rows:
                #    rows = sql_query(sql, db)
                
                print(rows['rows'])
                if not rows['rows']:
                    print("⚠️ No rows returned — retrying with instr() version...")
                    sql = convert_like_to_instr(sql)
                    rows = run_semantic_sql(sql, db, index, idmap)
        
                print(rows)
                row_list.append(rows)
        
        # unwrap rows
        query_rows = ""
        for i in range(len(row_list)):
            query_rows += f"Table {i} from {str(used_table_list[i])}:\n" + str(row_list[i]) + "\n"
        #print("query_rows:", query_rows)
        
        # use retrieved info as rag
        #first_ans = inference(question + "\n" + str(row_list))
        schema = ""
        for s in schema_queue:
            schema += s + "\n"
        print("---> GENERATOR PROMPT:", question + "\nSchema:" + schema + "\n" + query_rows)
        first_ans = inference(question + "\nSchema:" + schema + "\n" + query_rows).content
        
        if "Both" in mode:
            RAG_prompt = rephrase_question(question + "\n相關文件\n" + first_ans)
            print("---> RAG_prompt:", RAG_prompt)
            # store, score = route_query(RAG_prompt, rag_index, store_names) #can consider just using first_ans

            best_store, score = route_query(RAG_prompt, router_index, RAG_store_names)
            chosen_idx = RAG_store_names.index(best_store)
            index = RAG_stores[chosen_idx]

            # Load corresponding texts
            with open(f"vectorstores/{best_store}_texts.json", "r", encoding="utf-8") as f:
                texts = json.load(f)

            # Search inside selected store
            emb = np.array(
                client.embeddings.create(model=EMBED_MODEL, input="find recent RAG papers").data[0].embedding,
                dtype=np.float32
            ).reshape(1, -1)
            faiss.normalize_L2(emb)
            D, I = index.search(emb, k=3)
            context_string = ""
            for rank, idx in enumerate(I[0]):
                context_string = context_string + texts[idx] + "\n"

    
            conversations = [
                SystemMessage(content="You are a chatbot that integrates information retrieved from RAG into your answers. Use only relevant information from the retrieved chunks."),
                HumanMessage(content=RAG_prompt + "\n Contexts:\n" + context_string)
            ]

            output = llm.invoke(conversations).content



            # output = doc_RAG(RAG_prompt, store)
            #print(output)
            return output
        
        #sufficient_ans = check_ans(question + "\nAnswer: " + first_ans.content)
        #print("QUESTION ANSWERED:", sufficient_ans)
        #output = inference(question)
        #print(output)
              
        print("-------------------"*3)
    return first_ans

if __name__ == "__main__":
    inf_end_to_end()
