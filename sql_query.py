# sql_mapper_and_runner.py
import re
import sqlite3
from sqlglot import parse_one, exp
from rapidfuzz import process, fuzz

# ---------- CONFIG ----------
SQLITE_DB = "arxiv.db"   # path to your sqlite DB
MODEL_SQL = """SELECT *
FROM 契約測試資料20250923
WHERE 契約名稱 LIKE '%光學%';"""
FUZZY_THRESHOLD = 80  # 0-100, higher = stricter
MAX_ROWS = 500        # limit returned rows for safety
# ----------------------------

def get_tables(conn):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [r[0] for r in cur.fetchall()]

def get_columns(conn, table):
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return [r[1] for r in cur.fetchall()]

def normalize(s):
    return re.sub(r'\W+', '', s).lower()

def find_best_table(name, tables):
    # exact
    if name in tables: 
        return name
    # case-insensitive exact
    for t in tables:
        if t.lower() == name.lower():
            return t
    # normalized contains / contained
    n = normalize(name)
    for t in tables:
        tn = normalize(t)
        if n == tn or n in tn or tn in n:
            return t
    # fuzzy match
    match = process.extractOne(name, tables, scorer=fuzz.ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        return match[0]
    return None

def find_best_column(name, columns):
    # exact
    if name in columns:
        return name
    for c in columns:
        if c.lower() == name.lower():
            return c
    n = normalize(name)
    for c in columns:
        if n == normalize(c) or n in normalize(c) or normalize(c) in n:
            return c
    match = process.extractOne(name, columns, scorer=fuzz.ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        return match[0]
    return None

def parse_tables_columns(sql):
    """Return lists of table names and (table, column) Column expressions found by sqlglot."""
    try:
        tree = parse_one(sql, read='sqlite')
    except Exception as e:
        print("SQL parsing failed:", e)
        return [], []
    tables = []
    cols = []
    for t in tree.find_all(exp.Table):
        if isinstance(t.this, str):
            tables.append(t.this)
        else:
            # in some versions t.this can be an Identifier
            tables.append(t.this.name if hasattr(t.this, "name") else str(t.this))
    for c in tree.find_all(exp.Column):
        # Column expression often has .this = column name, .table = table identifier
        col_name = c.name
        tbl = None
        if c.table:
            tbl = c.table
        cols.append((tbl, col_name))
    return tables, cols

def rewrite_sql(sql, mapping):
    """Given mapping of original_name -> mapped_quoted_name, do safe replacements."""
    # sort keys longest-first to avoid partial replacement
    for orig in sorted(mapping.keys(), key=len, reverse=True):
        quoted = mapping[orig]
        # replace only whole-word occurrences (Unicode aware)
        pattern = r'(?<!["\w])' + re.escape(orig) + r'(?![\w"])'
        sql = re.sub(pattern, quoted, sql)
    return sql

def sql_query(model_sql=MODEL_SQL, db=SQLITE_DB):
    conn = sqlite3.connect(db, uri=False)
    tables = get_tables(conn)
    print("Tables in DB:", tables)

    # parse model SQL
    tbls_in_sql, cols_in_sql = parse_tables_columns(model_sql)
    print("Parsed tables:", tbls_in_sql)
    print("Parsed columns:", cols_in_sql)

    mapping = {}  # original identifier -> quoted replacement

    # Strategy 1: map tables
    for t in tbls_in_sql:
        if t is None:
            continue
        best = find_best_table(t, tables)
        if best:
            mapping[t] = f'"{best}"'
        else:
            # special heuristic: if there's a table that equals t + '_' + something, prefer that
            candidates = [x for x in tables if x.startswith(t + "_")]
            if candidates:
                mapping[t] = f'"{candidates[0]}"'
                print(f"Heuristic: mapped missing table {t} -> {candidates[0]}")
            else:
                print(f"Warning: no table match for '{t}'")

    # Strategy 2: map columns (table-aware)
    # Build table->columns cache
    table_columns = {tbl: get_columns(conn, tbl) for tbl in tables}

    for tbl, col in cols_in_sql:
        # If column already appears as fully-qualified like table_col that actually is a table name:
        if col in tables and (tbl not in tables):
            # model used a column that equals an actual table name (common when model mixes up)
            # interpret this as "use that table instead"
            mapping[col] = f'"{col}"'
            print(f"Detected column name that is actually a table: {col} -> using as table")
            continue

        if tbl:
            # try mapping using mapped table name if any
            mapped_tbl = mapping.get(tbl, None)
            real_tbl = None
            if mapped_tbl:
                # mapping stores quoted name -> remove quotes to get actual table
                real_tbl = mapping[tbl].strip('"')
            else:
                # best guess
                real_tbl = find_best_table(tbl, tables)
            if real_tbl:
                best_col = find_best_column(col, table_columns.get(real_tbl, []))
                if best_col:
                    mapping[col] = f'"{best_col}"'
                    # ensure table is quoted too
                    mapping[tbl] = f'"{real_tbl}"'
                else:
                    # if col looks like "T_col" where it includes a prefix "T_", try suffix
                    if '_' in col and col.startswith(real_tbl):
                        suffix = col.split('_', 1)[1]
                        best_col2 = find_best_column(suffix, table_columns.get(real_tbl, []))
                        if best_col2:
                            mapping[col] = f'"{best_col2}"'
                            mapping[tbl] = f'"{real_tbl}"'
                            print(f"Heuristic: mapped {col} -> column {best_col2} of table {real_tbl}")
                        else:
                            print(f"Warning: no column match for '{col}' in table '{real_tbl}'")
                    else:
                        print(f"Warning: no column match for '{col}' in table '{real_tbl}'")
            else:
                print(f"Warning: could not resolve table '{tbl}' for column '{col}'")
        else:
            # standalone column; try to find it in any table (prefer single match)
            found_tables = [t for t, cols in table_columns.items() if col in cols]
            if len(found_tables) == 1:
                mapping[col] = f'"{col}"'
            elif len(found_tables) > 1:
                # ambiguous: do nothing, require manual resolution
                print(f"Ambiguous column '{col}' found in multiple tables: {found_tables}")
            else:
                # try fuzzy across all columns
                all_cols = []
                for t, cols in table_columns.items():
                    all_cols.extend(cols)
                best = find_best_column(col, all_cols)
                if best:
                    mapping[col] = f'"{best}"'
                else:
                    print(f"Warning: no match for standalone column '{col}'")

    # Always also quote any table/column-like tokens detected by regex that haven't been mapped
    # Build mapping for raw table tokens (unqualified) found by simple regex from FROM clause if parse failed
    # For safety, ensure mapping keys are actual substrings present in original SQL
    mapping = {k: v for k, v in mapping.items() if k in model_sql}

    print("Identifier mapping:", mapping)
    new_sql = rewrite_sql(model_sql, mapping)

    # Safety: limit returned rows (wrap in SELECT * FROM (original) LIMIT MAX_ROWS)
    final_sql = f"SELECT * FROM ({new_sql.rstrip(';')}) LIMIT {MAX_ROWS};"
    rows = []

    print("Transformed SQL:")
    print(final_sql)
    # Validate with EXPLAIN QUERY PLAN
    try:
        cur = conn.cursor()
        cur.execute("EXPLAIN QUERY PLAN " + new_sql)
        plan = cur.fetchall()
        print("EXPLAIN QUERY PLAN:", plan)
    except Exception as e:
        print("EXPLAIN failed:", e)
        print("Do NOT execute the transformed SQL unless you review it.")
        conn.close()
        return

    # Execute and print first rows
    try:
        cur.execute(final_sql)
        rows = cur.fetchall()
        print(f"Returned {len(rows)} rows (showing up to {min(len(rows),10)}):")
        for r in rows[:10]:
            print(r)
    except Exception as e:
        print("Execution failed:", e)

    conn.close()
    return rows

if __name__ == "__main__":
    sql_query()

