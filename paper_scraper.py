import os
import time
import arxiv 
from datetime import datetime, timedelta, timezone
import ollama
import pandas as pd
import sys

MODEL = "gpt-oss:120b"     # Ollama model name
client = arxiv.Client(
    page_size=100,
    delay_seconds=3,
    num_retries=3,
)

client._session.verify = False

def generate_tags(title):
    """Use Ollama LLM to generate relevant tags for a paper title."""
    prompt = f"""
    You are a research paper tagger.
    Given the following paper title, generate 3-6 concise, general tags (comma-separated).
    The tags should reflect the paper's main topics, e.g. AI, Machine Translation, Chinese-English, NLP, etc.
    
    Title: "{title}"
    Tags:
    """
    try:
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
        tags = response["message"]["content"].strip()
        # Clean and normalize
        tags = tags.replace("\n", "").replace("Tags:", "").strip()
        return tags
    except Exception as e:
        print(f"Error generating tags for '{title}': {e}")
        return ""

def fetch_recent(category: str, max_results=20, days_back=3):
    """Fetch metadata of recent papers in category."""
    
    # Make cutoff timezone-aware (UTC)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    search = arxiv.Search(
        query=f"cat:{category}",
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=max_results
    )
    
    results = list(search.results())
    
    # r.published is already timezone-aware (UTC)
    recent = [r for r in results if r.published >= cutoff]
    
    return recent


def save_metadata(papers, out_csv="metadata.csv"):
    """Save metadata to CSV."""
    import csv
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","title","authors","published","categories","pdf_url"])
        for p in papers:
            print("title:", p.title)
            tags = generate_tags(p.title)
            #categories = f"{p.categories}, {tags}"
            lst = [item.strip() for item in tags.split(",")]
            categories = p.categories + lst
            print("final categories:", categories)
            writer.writerow([
                p.entry_id,
                p.title.replace("\n"," "),
                ";".join(author.name for author in p.authors),
                p.published.isoformat(),
                ";".join(categories),
                p.pdf_url
            ])

def download_papers(papers, folder="downloads"):
    os.makedirs(folder, exist_ok=True)
    for p in papers:
        fname = p.title + ".pdf"
        # sanitize filename if needed
        fpath = os.path.join(folder, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {p.title!r} → {fname}")
            p.download_pdf(dirpath=folder, filename=fname)
            time.sleep(3)  # respectful delay
        else:
            print(f"Already downloaded {fname}")



def csv_to_xlsx(csv_file, xlsx_file=None):
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return

    if xlsx_file is None:
        xlsx_file = os.path.splitext(csv_file)[0] + ".xlsx"

    print(f"📄 Reading CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    print(f"💾 Saving Excel: {xlsx_file}")
    df.to_excel(xlsx_file, index=False, engine="openpyxl")

    print("✅ Conversion complete!")


if __name__ == "__main__":
    category = "cs.AI"  # change as needed
    papers = fetch_recent(category, max_results=10, days_back=7)
    print(f"Found {len(papers)} new papers in {category}")
    save_metadata(papers, out_csv="recent_arxiv.csv")
    download_papers(papers, folder="recent_arxiv_cs_AI")
    csv_to_xlsx("recent_arxiv.csv", "recent_arxiv.xlsx")
    

