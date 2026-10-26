"""
RAG Search Assignment - Chunking, Ingestion and Semantic Search with Filtering

What this program does:
1. Takes all files from data_sources/ folder and splits them into chunks
2. Saves each chunk with "source" metadata (so I know which file it came from)
3. Lets me search only in a specific file using where={"source": "filename"}
4. Sends the search results to an LLM agent that answers with file name and page number
"""

import os
import re
from pathlib import Path

import chromadb
import pymupdf  # For reading PDF files (pip install pymupdf)
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load API key from .env file
load_dotenv()

# ============== CONFIGURATION ==============
# Path(__file__).parent = folder where THIS script is located
DATA_SOURCES_DIR = Path(__file__).parent / "data_sources"

# Chunk settings - these control how text is split
# CHUNK_SIZE: how many characters per chunk (500 is a good balance)
# CHUNK_OVERLAP: characters that repeat between chunks (so I don't cut sentences in half)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Embedding model - this transforms text into numbers (vectors)
# all-MiniLM-L6-v2 outputs 384 dimensions, it's small and fast
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Collection name = like a table name in a database
COLLECTION_NAME = "rag_documents"

# ============== OPENROUTER CLIENT ==============
# This connects to OpenRouter API (works like OpenAI but with many models)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY")
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Splits a long text into smaller pieces (chunks).

    - LLMs have token limits
    - Embeddings work better on shorter text
    - Search is more precise with smaller pieces

    Returns a list of dicts with: text, page number (estimated), chunk index
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end].strip()

        if chunk_content:
            # Estimate page number based on position
            # Assuming ~2000 characters per page
            estimated_page = (start // 2000) + 1

            # If the text has "Page X:" markers, use those instead
            page_match = re.search(r'Page\s+(\d+)', chunk_content)
            if page_match:
                estimated_page = int(page_match.group(1))

            chunks.append({
                "text": chunk_content,
                "page": estimated_page,
                "chunk_index": chunk_index
            })
            chunk_index += 1

        # Move forward, but keep some overlap so sentences aren't cut
        start += chunk_size - overlap

    return chunks


def load_and_chunk_files(data_dir: Path) -> list[dict]:
    """
    Reads all files from data_sources/ and chunks them.

    Important: adds "source" metadata to each chunk!
    This is how I can later filter: "search only in paragraphs.txt"

    Supports: .txt, .md, and .pdf files
    """
    all_chunks = []

    for file_path in data_dir.glob("*.*"):
        suffix = file_path.suffix.lower()
        print(f"Processing: {file_path.name}")

        # Handle different file types
        if suffix == ".pdf":
            # PDF files - use pymupdf to extract text
            doc = pymupdf.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

        elif suffix in [".txt", ".md"]:
            # Text and markdown files - read directly
            text = file_path.read_text(encoding="utf-8")

        else:
            # Skip unsupported file types
            print(f"  -> Skipping (unsupported format)")
            continue

        chunks = chunk_text(text)

        # THIS IS KEY: add source filename to each chunk's metadata
        for chunk in chunks:
            chunk["source"] = file_path.name

        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks created")

    return all_chunks


def ingest_to_chromadb(chunks: list[dict], model: SentenceTransformer) -> chromadb.Collection:
    """
    Takes chunks and saves them to ChromaDB (vector database).

    Steps:
    1. Generate embeddings (text -> list of 384 numbers)
    2. Save to ChromaDB with: id, embedding, document text, metadata

    Metadata includes "source" and "page" - needed for filtering and citing
    """
    print("\nGenerating embeddings...")
    texts = [chunk["text"] for chunk in chunks]

    # This is where the magic happens - text becomes numbers
    # normalize_embeddings=True makes all vectors length 1 (better for comparison)
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    print("Saving to ChromaDB...")

    # PersistentClient saves to disk (survives restart)
    client_db = chromadb.PersistentClient(path="./chroma_db")

    # Delete old collection if exists (for clean reload)
    try:
        client_db.delete_collection(name=COLLECTION_NAME)
    except:
        pass

    # Create collection with cosine similarity (standard for text)
    collection = client_db.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data for insertion
    # ID format: "filename.txt:chunk_0" - unique identifier
    ids = [f"{chunk['source']}:chunk_{chunk['chunk_index']}" for chunk in chunks]
    metadatas = [{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks]
    documents = texts

    # Upsert = insert or update if exists
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Loaded {len(chunks)} chunks into collection '{COLLECTION_NAME}'")
    return collection


def get_available_sources(collection: chromadb.Collection) -> list[str]:
    """
    Returns list of all source files in the database.
    Useful to show user what files they can search in.
    """
    all_data = collection.get(include=["metadatas"])
    sources = set()
    for metadata in all_data["metadatas"]:
        if metadata and "source" in metadata:
            sources.add(metadata["source"])
    return sorted(list(sources))


def search_in_source(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query: str,
    source: str,
    n_results: int = 3,
    max_distance: float=0.5
) -> dict:
    """
    Searches in the database BUT only in chunks from a specific file.

    The key part is: where={"source": source}
    This filters results to only show chunks from that file!

    Returns top n_results most similar chunks.
    """
    # Convert my question to embedding (same format as stored chunks)
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # THIS IS THE FILTERED SEARCH - only searches in one file
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"source": source},  # <-- FILTER BY SOURCE FILE
        include=["documents", "metadatas", "distances"]
    )
    # Filtreaza rezultatele cu distanta prea mare
    filtered_docs = []
    filtered_metas = []
    filtered_dists = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if dist < max_distance: #pastreaza doar daca e sub limita
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            filtered_dists.append(dist)
        #inlocuieste rezultatele cu cele filtrate (tema 3.2)
        results["documents"] = [filtered_docs]
        results["metadatas"] = [filtered_metas]
        results["distances"] = [filtered_dists]


    return results


def ask_llm_agent(query: str, source: str, search_results: dict) -> str:
    """
    Takes search results and asks LLM to answer the question.

    The system prompt tells the LLM:
    - Only use the provided context (don't make stuff up)
    - Always mention file name and page number
    - Say if information wasn't found
    """
    # Format the search results nicely for the LLM
    context_parts = []
    for i, (doc, metadata, distance) in enumerate(zip(
        search_results["documents"][0],
        search_results["metadatas"][0],
        search_results["distances"][0]
    )):
        # Distance is 0-1, lower = more similar
        # Relevance = 1 - distance (higher = more relevant)
        context_parts.append(
            f"[Fragment {i+1}] (Page {metadata['page']}, Relevance: {1-distance:.2f})\n{doc}"
        )

    context = "\n\n".join(context_parts)

    # System prompt - instructions for the LLM
    # Written in English for better precision, but can be Romanian too
    system_prompt = """You are an assistant that answers questions based ONLY on the provided context.

RULES:
1. Answer ONLY based on information in the context
2. ALWAYS mention the source file name and page number
3. If the information is not in the context, clearly say you didn't find it
4. Be concise and precise
5. Respond in the same language as the question

RESPONSE FORMAT:
- Your answer
- Source: [filename]
- Page: [page number]
"""

    user_prompt = f"""Question: {query}

Source file: {source}

Context found:
{context}

Answer the question using the context above."""

    # Send to LLM and get response
    response = client.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    ).choices[0].message.content

    return response


def main():
    """
    Main function - runs the whole RAG pipeline:
    1. Load embedding model
    2. Check if database exists, if not create it
    3. Interactive loop: user asks questions, I search and answer
    """

    print("=" * 60)
    print("RAG SEARCH - Semantic Search with Source Filtering")
    print("=" * 60)

    # Step 1: Load the embedding model (this takes a few seconds first time)
    print("\n[1] Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Step 2: Check if database already has data
    client_db = chromadb.PersistentClient(path="./chroma_db")

    try:
        collection = client_db.get_collection(name=COLLECTION_NAME)
        print(f"\n[2] Collection '{COLLECTION_NAME}' already exists.")

        reingest = input("Do you want to reload files? (yes/no): ").strip().lower()
        if reingest in ["yes", "da", "y"]:
            raise Exception("Force reingest")
    except:
        # No collection exists, need to create it
        print(f"\n[2] Chunking and ingesting files from '{DATA_SOURCES_DIR}'...")
        chunks = load_and_chunk_files(DATA_SOURCES_DIR)

        if not chunks:
            print("ERROR: No files found in data_sources/!")
            print("Add some .txt or .md files there first.")
            return

        collection = ingest_to_chromadb(chunks, model)

    # Step 3: Show what files are available
    sources = get_available_sources(collection)
    print(f"\n[3] Available files: {sources}")

    # Step 4: Interactive search loop
    print("\n" + "=" * 60)
    print("Enter your query in format: filename.txt, your question")
    print("Example: paragraphs.txt, Who is Maria?")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        user_input = input("\nYou> ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Parse input: expect "filename, question"
        if "," not in user_input:
            print("Invalid format! Use: filename.txt, your question")
            continue

        parts = user_input.split(",", 1)
        source_file = parts[0].strip()
        question = parts[1].strip()

        # Check if file exists in database
        if source_file not in sources:
            print(f"ERROR: File '{source_file}' not in database.")
            print(f"Available files: {sources}")
            continue

        # Step 5: Search (filtered by source)
        print(f"\nSearching in '{source_file}'...")
        results = search_in_source(collection, model, question, source_file)
        print("Distante gasite: ", results["distances"])

        #tema 3.2
        if not results["documents"][0]:
            print("Nu s-a gasit nimic relevant (distanta e prea mare).")
            continue


        if not results["documents"][0]:
            print("No relevant results found.")
            continue

        # Step 6: Ask LLM to answer
        print("Asking LLM agent...")
        answer = ask_llm_agent(question, source_file, results)

        print("\n" + "-" * 40)
        print("AGENT RESPONSE:")
        print("-" * 40)
        print(answer)


if __name__ == "__main__":
    main()
