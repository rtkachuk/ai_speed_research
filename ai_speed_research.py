"""RAG command line app using LiteLLM and a choice of vector store.

Usage:
    python ai_speed_research.py                  (interactive menu)
    python ai_speed_research.py ingest <path> [<path> ...]
    python ai_speed_research.py ask "<question>"
    python ai_speed_research.py chat
    python ai_speed_research.py list
    python ai_speed_research.py models
    python ai_speed_research.py reset

Environment:
    LITELLM_HOST      LiteLLM base URL
    LITELLM_API_KEY   LiteLLM API key (required)
    EMBED_MODEL       Embedding model (default: openai/nomic-embed-text)
    CHAT_MODEL        Chat model (default: openai/llama3.2)

    VECTOR_STORE      chroma, pgvector or qdrant. If unset, a menu asks which
                      to use; without a terminal, chroma is used.
    PGVECTOR_DSN      Postgres DSN
    QDRANT_URL        Qdrant URL
    QDRANT_API_KEY    Qdrant API key (optional)

Example:
    VECTOR_STORE=qdrant LITELLM_API_KEY=sk-... python ai_speed_research.py chat
"""

import glob
import itertools
import os
import shlex
import sys
import textwrap
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import litellm
from litellm import completion, embedding

LITELLM_HOST = os.environ.get("LITELLM_HOST", "")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "openai/nomic-embed-v2")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "openai/llama-3.2-3b")

VECTOR_STORE = os.environ.get("VECTOR_STORE")
PGVECTOR_DSN = os.environ.get("PGVECTOR_DSN", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

DEFAULT_STORE = "chroma"
CHROMA_PATH = "./chroma_db"
COLLECTION = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 20
BATCH_SIZE = 64
QDRANT_SCROLL_SIZE = 1000

Vector = list[float]


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    position: int

    @property
    def id(self) -> str:
        return f"{Path(self.source).stem}__{self.position}"


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    def search(self, vector: Vector, limit: int) -> list[Chunk]: ...

    def count(self) -> int: ...

    def chunks_per_source(self) -> Counter[str]: ...

    def delete(self) -> bool:
        """Remove all stored data. Return False if there was nothing to remove."""
        ...

class ChromaStore:
    def __init__(self) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=CHROMA_PATH)

    def _collection(self):
        return self._client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        self._collection().upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=list(vectors),
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"source": c.source, "chunk": c.position} for c in chunks],
        )

    def search(self, vector: Vector, limit: int) -> list[Chunk]:
        collection = self._collection()
        results = collection.query(
            query_embeddings=[vector],
            n_results=min(limit, collection.count()),
        )
        return [
            Chunk(text, meta["source"], meta["chunk"])
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def count(self) -> int:
        return self._collection().count()

    def chunks_per_source(self) -> Counter[str]:
        metadatas = self._collection().get(include=["metadatas"])["metadatas"]
        return Counter(meta["source"] for meta in metadatas)

    def delete(self) -> bool:
        from chromadb.errors import NotFoundError

        try:
            self._client.delete_collection(COLLECTION)
        except NotFoundError:
            return False
        return True


class PgVectorStore:
    def __init__(self) -> None:
        import psycopg

        self._conn = psycopg.connect(PGVECTOR_DSN, autocommit=True)

    def _table_exists(self) -> bool:
        row = self._conn.execute("SELECT to_regclass('documents')").fetchone()
        return row[0] is not None

    def _create_schema(self, dimensions: int) -> None:
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Postgres cannot bind DDL parameters; dimensions is always an int.
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS documents (
                id text PRIMARY KEY,
                content text NOT NULL,
                source text NOT NULL,
                chunk integer NOT NULL,
                embedding vector({dimensions:d}) NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING hnsw (embedding vector_cosine_ops)
            """
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        self._create_schema(len(vectors[0]))
        rows = [
            (chunk.id, chunk.text, chunk.source, chunk.position, str(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with self._conn.transaction(), self._conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO documents (id, content, source, chunk, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    source = EXCLUDED.source,
                    chunk = EXCLUDED.chunk,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )

    def search(self, vector: Vector, limit: int) -> list[Chunk]:
        rows = self._conn.execute(
            """
            SELECT content, source, chunk
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (str(vector), limit),
        ).fetchall()
        return [Chunk(*row) for row in rows]

    def count(self) -> int:
        if not self._table_exists():
            return 0
        return self._conn.execute("SELECT count(*) FROM documents").fetchone()[0]

    def chunks_per_source(self) -> Counter[str]:
        if not self._table_exists():
            return Counter()
        rows = self._conn.execute(
            "SELECT source, count(*) FROM documents GROUP BY source"
        ).fetchall()
        return Counter(dict(rows))

    def delete(self) -> bool:
        if not self._table_exists():
            return False
        self._conn.execute("DROP TABLE documents")
        return True


class QdrantStore:
    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    def _exists(self) -> bool:
        return self._client.collection_exists(COLLECTION)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        from qdrant_client import models

        if not self._exists():
            self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]),
                    distance=models.Distance.COSINE,
                ),
            )
        # Qdrant point IDs must be integers or UUIDs.
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk": chunk.position,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=COLLECTION, points=points, wait=True)

    def search(self, vector: Vector, limit: int) -> list[Chunk]:
        response = self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            Chunk(p.payload["text"], p.payload["source"], p.payload["chunk"])
            for p in response.points
        ]

    def count(self) -> int:
        if not self._exists():
            return 0
        return self._client.count(COLLECTION, exact=True).count

    def chunks_per_source(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        if not self._exists():
            return counts

        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=COLLECTION,
                limit=QDRANT_SCROLL_SIZE,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            counts.update(point.payload["source"] for point in points)
            if offset is None:
                return counts

    def delete(self) -> bool:
        if not self._exists():
            return False
        self._client.delete_collection(COLLECTION)
        return True


STORES: dict[str, type[VectorStore]] = {
    "chroma": ChromaStore,
    "pgvector": PgVectorStore,
    "qdrant": QdrantStore,
}


def prompt_for_store(default: str = DEFAULT_STORE) -> str:
    names = list(STORES)
    print("Select a vector store:")
    for number, name in enumerate(names, start=1):
        print(f"  {number}) {name}")

    choices = {str(number): name for number, name in enumerate(names, start=1)}
    choices |= {name: name for name in names}
    choices[""] = default
    while True:
        try:
            answer = input(f"Choice [{default}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nNo vector store selected.")
        if answer in choices:
            return choices[answer]
        print(f"Enter a number from 1 to {len(names)}, or a store name.")


def choose_store() -> str:
    if VECTOR_STORE:
        if VECTOR_STORE not in STORES:
            sys.exit(f"Unknown VECTOR_STORE {VECTOR_STORE!r}. Choose from: {', '.join(STORES)}")
        return VECTOR_STORE
    if not sys.stdin.isatty():
        return DEFAULT_STORE
    return prompt_for_store()


def setup_litellm() -> None:
    if not LITELLM_API_KEY:
        raise ValueError("LITELLM_API_KEY is required but not set")

    litellm.api_base = LITELLM_HOST
    litellm.api_key = LITELLM_API_KEY
    litellm.set_verbose = False
    litellm.ssl_verify = False


def server_label() -> str:
    return LITELLM_HOST.rstrip("/")


def check_server(store_name: str) -> None:
    try:
        completion(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            api_base=LITELLM_HOST,
            api_key=LITELLM_API_KEY,
        )
    except Exception as exc:
        raise ConnectionError(
            f"Cannot reach LiteLLM at {LITELLM_HOST}: {exc}\n"
            "Check LITELLM_HOST and LITELLM_API_KEY."
        ) from exc

    print(
        f"Connected to LiteLLM at {server_label()} "
        f"(embed={EMBED_MODEL}, chat={CHAT_MODEL}, store={store_name})"
    )


def embed(texts: Sequence[str]) -> list[Vector]:
    response = embedding(
        model=EMBED_MODEL,
        input=list(texts),
        api_base=LITELLM_HOST,
        api_key=LITELLM_API_KEY,
    )
    return [item["embedding"] for item in response.data]


def ask_llm(prompt: str) -> str:
    response = completion(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_base=LITELLM_HOST,
        api_key=LITELLM_API_KEY,
    )
    return response.choices[0].message.content.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    step = size - overlap
    pieces = (text[start:start + size].strip() for start in range(0, len(text), step))
    return [piece for piece in pieces if piece]


def load_chunks(path: str) -> list[Chunk]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return [Chunk(piece, path, i) for i, piece in enumerate(chunk_text(text))]


def find_files(paths: Sequence[str]) -> list[str]:
    files: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "**/*.txt"), recursive=True))
            files.extend(glob.glob(os.path.join(path, "**/*.md"), recursive=True))
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"Not found: {path}", file=sys.stderr)
    return files


def build_prompt(question: str, hits: Sequence[Chunk]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {hit.source}, chunk {hit.position}]\n{hit.text}" for hit in hits
    )
    return (
        "You are a helpful assistant. Answer ONLY from the provided context.\n"
        "If the answer isn't there, say so - do not invent information.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def answer_question(store: VectorStore, question: str) -> tuple[list[Chunk], str, str]:
    """Return the retrieved chunks, the model's answer and a timing summary."""
    started = time.perf_counter()
    vector = embed([question])[0]
    embedded = time.perf_counter()
    hits = store.search(vector, TOP_K)
    searched = time.perf_counter()
    answer = ask_llm(build_prompt(question, hits))
    finished = time.perf_counter()

    timing = (
        f"Time: prompt embedding {embedded - started:.3f}s | "
        f"RAG search {searched - embedded:.3f}s | "
        f"model {finished - searched:.3f}s | "
        f"total {finished - started:.3f}s"
    )
    return hits, answer, timing


def print_answer(question: str, hits: Sequence[Chunk], answer: str) -> None:
    separator = "-" * 60
    print(f"\n{question}\n\n{separator}")
    print(textwrap.fill(answer, width=72))
    print(f"{separator}\n\nSources:")
    for hit in hits:
        snippet = hit.text[:80].replace("\n", " ")
        print(f"  {hit.source} (chunk {hit.position}): {snippet}...")


def open_populated_store(store_name: str) -> VectorStore | None:
    store = STORES[store_name]()
    if store.count() == 0:
        print("Vector store is empty. Ingest some documents first.")
        return None
    return store


def cmd_ingest(store_name: str, paths: Sequence[str]) -> None:
    files = find_files(paths)
    if not files:
        print("No .txt or .md files found.")
        return

    check_server(store_name)
    store = STORES[store_name]()
    total = 0
    for path in files:
        chunks = load_chunks(path)
        for batch in itertools.batched(chunks, BATCH_SIZE):
            store.upsert(batch, embed([chunk.text for chunk in batch]))
        print(f"  {path}: {len(chunks)} chunks")
        total += len(chunks)

    print(f"\nIngested {len(files)} file(s), {total} chunks into {store_name}")


def cmd_list(store_name: str) -> None:
    counts = STORES[store_name]().chunks_per_source()
    if not counts:
        print("Vector store is empty. Ingest some documents first.")
        return

    print(f"\n{counts.total()} chunks from {len(counts)} file(s) in {store_name}:\n")
    for source, count in sorted(counts.items()):
        print(f"  {source} ({count} chunks)")


def cmd_models(store_name: str | None) -> None:
    print(f"\nLiteLLM at {server_label()}\n")
    print(f"  chat:  {CHAT_MODEL}")
    print(f"  embed: {EMBED_MODEL}")
    print(f"  store: {store_name or 'chosen from the menu'}")
    print("\nSet CHAT_MODEL, EMBED_MODEL and VECTOR_STORE to change these.")


def cmd_ask(store_name: str, question: str) -> None:
    check_server(store_name)
    store = open_populated_store(store_name)
    if store is None:
        return
    hits, answer, timing = answer_question(store, question)
    print_answer(question, hits, answer)
    print(f"\n{timing}")


def cmd_chat(store_name: str) -> None:
    check_server(store_name)
    store = open_populated_store(store_name)
    if store is None:
        return

    print(f"\nRAG chat [{CHAT_MODEL} at {server_label()}, store={store_name}]")
    print("Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            return
        _, answer, timing = answer_question(store, question)
        print(f"\nAssistant: {textwrap.fill(answer, width=72)}\n\n{timing}\n")


def cmd_reset(store_name: str) -> None:
    if STORES[store_name]().delete():
        print(f"Deleted '{COLLECTION}' from {store_name}.")
    else:
        print("Nothing to delete.")


def menu_ingest(store_name: str) -> None:
    answer = input("Paths to ingest (quote any with spaces): ")
    paths = [os.path.expanduser(path) for path in shlex.split(answer)]
    if paths:
        cmd_ingest(store_name, paths)


def menu_ask(store_name: str) -> None:
    question = input("Question: ").strip()
    if question:
        cmd_ask(store_name, question)


def menu_reset(store_name: str) -> None:
    answer = input(f"Delete all documents from {store_name}? [y/N]: ")
    if answer.strip().lower() in {"y", "yes"}:
        cmd_reset(store_name)
    else:
        print("Cancelled.")


MENU_ACTIONS = {
    "1": ("Ingest files", menu_ingest),
    "2": ("Ask a question", menu_ask),
    "3": ("Chat", cmd_chat),
    "4": ("List documents", cmd_list),
    "5": ("Show models", cmd_models),
    "6": ("Reset vector store", menu_reset),
}
MENU_SWITCH_STORE = "7"
MENU_EXIT = "0"


def print_menu(store_name: str) -> None:
    print(f"\nRAG menu (store: {store_name})")
    for key, (label, _) in MENU_ACTIONS.items():
        print(f"  {key}) {label}")
    print(f"  {MENU_SWITCH_STORE}) Switch vector store")
    print(f"  {MENU_EXIT}) Exit")


def run_menu() -> None:
    store_name = choose_store()
    while True:
        print_menu(store_name)
        try:
            choice = input("Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice == MENU_EXIT:
            return
        if choice == MENU_SWITCH_STORE:
            store_name = prompt_for_store(default=store_name)
            continue
        if choice not in MENU_ACTIONS:
            print(f"Unknown option: {choice!r}")
            continue

        _, action = MENU_ACTIONS[choice]
        try:
            action(store_name)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
        # Any failure in one action should leave the menu usable.
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)


def run_command(command: str, rest: Sequence[str]) -> None:
    if command == "models":
        cmd_models(VECTOR_STORE)
        return
    if command == "ingest" and not rest:
        print("Provide at least one path.")
        return
    if command == "ask" and not rest:
        print('Usage: ask "<question>"')
        return

    commands = {
        "ingest": lambda store_name: cmd_ingest(store_name, rest),
        "ask": lambda store_name: cmd_ask(store_name, " ".join(rest)),
        "chat": cmd_chat,
        "list": cmd_list,
        "reset": cmd_reset,
    }
    if command not in commands:
        print(f"Unknown command: {command}\n\n{__doc__}")
        return
    commands[command](choose_store())


def main() -> None:
    setup_litellm()

    args = sys.argv[1:]
    if not args:
        if sys.stdin.isatty():
            run_menu()
        else:
            print(__doc__)
        return

    command, *rest = args
    try:
        run_command(command, rest)
    except ConnectionError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()