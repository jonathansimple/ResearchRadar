# ResearchRadar
## Introduction
**ResearchRadar** is an automated research-monitoring system designed to keep you updated on the latest academic work with minimal effort. The backend pipeline continuously scrapes arXiv for papers in selected categories and time windows, downloads their PDFs, and compiles structured metadata into a searchable spreadsheet. This creates a unified local corpus of both **full-text documents** and **tabular publication data**, ready for intelligent retrieval.

At the core of ResearchRadar is an **agent-based architecture** with a **planner** that decides — per query — which retrieval strategy to invoke. Document-oriented questions (e.g., “What are the recent papers on RAG?” or “What are the key innovations in paper X?”) are routed through a RAG agent that reasons over PDFs, while quantitative or trend-based queries (e.g., “How many CV papers were published in the last 3 months?”) are handled by a text-to-SQL agent. Both agents rely on **Chain-of-Thought reasoning** to validate SQL generation, ensure answer completeness, and self-check the soundness of retrieved results. A conversational chatbot interface ties everything together, giving users a natural way to explore research trends and the evolving academic landscape.

## Setting Up
To run the repo, first create a conda environment
```
conda create -n rradar python=3.10
conda activate rradar
pip install hipporag
pip install -r requirements.txt
```

## Scraper Module
The paper scraper will generate a folder and csv/xlsx file based on the specified arxiv categories and time range. Due to the volume of submissions on arvix, there is also a limit on how many papers are downloaded and logged. Additionally, in adherence to the arxiv API usage rules, there is a sleep timer within the program that waits for 3 seconds before beginning a new download.

```
python paper_scraper.py
```

After running this program you should have two spreadsheets containing the metadata of the collected papers in .csv and .xlsx format and a folder containing the downloaded PDFs.

## Building the SQL and RAG data stores
Before the raw documents are ready to be used for retrieval, they must be preprocessed.

For the text-to-sql pipeline, the SQL database is prepared using the following scripts
```
python excel_to_sql.py
python gen_embeddings.py
```

For the RAG pipeline, the dataset is prepared using the following scripts
```
python pdf_to_json.py
python create_doc_rag_index.py
```

After running these programs, you should have a folder containing the converted json papers, files for the SQL table, and a faiss index for the download papers.

## Front End Chatbot
To launch the chatbot, run
```
python paper_gradio_interface.py
```
The default page for the gradio application is 0.0.0.0:7860.
