# ai_speed_research

A small command line RAG app which could be used to compare vector databases. It ingests text files, answers questions about them through a LiteLLM server, and prints how long each step took.

It works with ChromaDB, pgvector and Qdrant. Everything else stays the same between them, so the timings are comparable.

## Setup

You need Python 3.12 or newer and a LiteLLM server with one chat model and one embedding model.

```bash
python3 -m venv dev
source dev/bin/activate
pip install -r requirements.txt
```

ChromaDB runs inside the script and saves to `./chroma_db`. The other two need a server, and Docker is the easiest way to get one:

```bash
docker run -d --name rag-pgvector \
  -e POSTGRES_PASSWORD=changeme -e POSTGRES_DB=rag \
  -p 127.0.0.1:5432:5432 \
  -v rag-pgvector-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17

docker run -d --name rag-qdrant \
  -p 127.0.0.1:6333:6333 \
  -v rag-qdrant-data:/qdrant/storage \
  qdrant/qdrant
```

## Configuration

The script reads these environment variables. The defaults in the code point at my own server, so set the first four if you're running this anywhere else.

| Variable | What it is |
|---|---|
| `LITELLM_HOST` | LiteLLM base URL |
| `LITELLM_API_KEY` | LiteLLM API key |
| `EMBED_MODEL` | Embedding model, e.g. `openai/nomic-embed-text` |
| `CHAT_MODEL` | Chat model, e.g. `openai/llama3.2` |
| `VECTOR_STORE` | `chroma`, `pgvector` or `qdrant`. Skips the database menu |
| `PGVECTOR_DSN` | e.g. `postgresql://postgres:changeme@localhost:5432/rag` |
| `QDRANT_URL` | Defaults to `http://localhost:6333` |
| `QDRANT_API_KEY` | Only if your Qdrant uses one |

## Usage

Run it with no arguments for the menu:

```bash
python ai_speed_research.py
```

Or call a command directly:

```bash
python ai_speed_research.py ingest sampletext.txt
python ai_speed_research.py ask "What do you know about ...?"
python ai_speed_research.py chat
python ai_speed_research.py list
python ai_speed_research.py models
python ai_speed_research.py reset
```

`ingest` takes files or directories. Directories are searched recursively for `.txt` and `.md` files.

If `VECTOR_STORE` isn't set, you get asked which database to use. When there's no terminal (cron, pipes) it uses Chroma.

Careful with `reset`: from the command line it deletes everything in the selected database without asking. The menu version asks first.

## Reading the timings

Every answer ends with a line like this:

```
Time: prompt embedding 0.105s | RAG search 0.205s | model 0.304s | total 0.614s
```

- **prompt embedding** is the time to embed the question.
- **RAG search** is the query to the database.
- **model** is the time to generate the full answer.
- **total** is all three added up.

The model usually takes most of the time and varies a lot, so look at `RAG search` when comparing databases, and use more than one question. Chroma runs in-process and the other two go over the network, which gives it a small head start.

### Functions

- Loads `.txt` and `.md` files and splits them into chunks.
- Stores embeddings in ChromaDB, PostgreSQL with pgvector, or Qdrant.
- Finds relevant chunks and generates an answer with a language model through LiteLLM.
- Measures the time spent on query embedding, search and the model.
- Runs from an interactive menu or from individual commands.

### Third-party components

These libraries are not part of the work and are used under their own licenses:

| Library | License |
|---|---|
| litellm | MIT |
| chromadb | Apache-2.0 |
| qdrant-client | Apache-2.0 |
| psycopg | LGPL-3.0 |
