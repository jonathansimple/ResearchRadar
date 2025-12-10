import pandas as pd
from sqlalchemy import create_engine

# Path to your Excel file
excel_file = "recent_arxiv.xlsx"

# Read the Excel sheet(s)
df = pd.read_excel(excel_file, sheet_name="Sheet1")  # or use None to read all sheets

# Connect to SQLite (creates db file if not exists)
engine = create_engine("sqlite:///arxiv.db")

# Write dataframe to SQL table
df.to_sql("arxiv_index", con=engine, if_exists="replace", index=False)

print("Data imported into SQL successfully!")

