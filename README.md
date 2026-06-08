# Intelligent RAG-Powered Document Chatbot

A production-ready Retrieval-Augmented Generation (RAG) system built to ingest localized PDF documentation, parse complex text into semantic chunks, and provide highly accurate context-isolated answers.

## 🚀 Key Features

- **Deterministic Document Ingestion:** Automated ingestion pipeline that extracts text from unstructured PDF formats cleanly.
- **Semantic Text Chunking:** Employs optimized sliding-window recursive parsing to preserve paragraph and data continuity before vector generation.
- **Persistent Vector Embedding Storage:** Uses ChromaDB as a localized database on disk to prevent duplicate embedding compute overhead across user sessions.
- **Hallucination Mitigation:** Enforces rigid prompt-engineering guardrails to isolate LLM reasoning explicitly to the document context.
- **Clean Interface:** Designed with zero external latency constraints for rapid contextual querying.

## 🏗️ Architecture Flow

1. **Document Loading:** Native PDF raw character extraction.
2. **Text Processing:** Chunk segmentation with structured token constraints and overlaps.
3. **Vector Matrix Generation:** Map text chunks into dense semantic spaces.
4. **Storage:** ChromaDB collection lookup and persistence mapping.
5. **Retrieval Strategy:** Top-K nearest-neighbor cosine similarity matching on user queries.
6. **Generation:** Context compilation and localized prompt delivery via Cohere API.

## 🛠️ Tech Stack

- **Primary Language:** Python 3.10+
- **Vector Database:** ChromaDB
- **LLM Engine:** Cohere API (Command Series)
- **Document Utilities:** PyPDF / LangChain Document Parsers
- **Environment Management:** python-dotenv

## 📋 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mrlucky5123/MINI-PROJECT-.git
cd MINI-PROJECT-
```

### 2. Establish Virtual Environment & Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory and add your API credentials:

```bash
COHERE_API_KEY=your_secret_cohere_api_key_here
CHROMA_DB_PATH=./chroma_db
```

### 4. Execute Ingestion Pipeline

Place your source PDF files inside the `data` directory and run:

```bash
python ingest.py
```

### 5. Initialize Chatbot Application

```bash
python chatbot.py
```

## ⚠️ Troubleshooting

If you encounter any issues during setup or while running the project, please reach out for support.

---

**Happy chatting! 🤖**
