# ai-rag-pipeline-supabase

This project implements a Retrieval-Augmented Generation (RAG) pipeline with multiple variations, leveraging Langchain, HuggingFace, Chroma, and Supabase.

## Architecture Highlights
- **Document Loading:** Reads `.txt` files from a local `docs/` directory.
- **Embeddings:** Uses local HuggingFace sentence transformers (`all-MiniLM-L6-v2`) for embeddings.
- **LLM:** Uses `Qwen/Qwen2-72B-Instruct` via `HuggingFaceEndpoint` for response generation.
- **Vector Stores:** Supports both local ChromaDB and cloud-based Supabase (pgvector).
- **Chunking Strategies:** Includes standard character-based chunking and advanced **semantic chunking**.
- **History-Aware Generation:** Includes a chat interface that remembers conversation history and rephrases questions contextually.

## Key Files

### `1_ingestion_pipeline.py`
A baseline ingestion script that:
1. Loads text documents from the `docs/` folder.
2. Splits them into chunks using a `CharacterTextSplitter`.
3. Creates embeddings and stores them locally using ChromaDB in `db/chroma_db`.

### `1C_ingestion_semantic_chunking.py`
An advanced ingestion script that:
1. Loads documents from the `docs/` folder.
2. Uses Langchain's `SemanticChunker` to intelligently group text by semantic meaning rather than arbitrary character counts.
3. Stores embeddings in a **Supabase** database using `pgvector`. This requires setting `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` `.env` variables.

### `4A_history_aware_generation_supabase.py`
A comprehensive RAG interaction script that:
1. Connects to the Supabase vector store populated by the ingestion scripts.
2. Provides a conversational loop (`start_chat`).
3. Takes conversation history into account, rewriting follow-up queries so they are self-contained.
4. Retrieves relevant context and generates an answer using the Qwen model.
*Note: Contains a patch for LangChain's Supabase vector store integration to properly handle filtering with newer Supabase client versions.*

### `schema.sql`
A SQL script containing the expected database schema for Supabase:
- Enables the pgvector extension.
- Creates the `documents` table (`id`, `content`, `metadata`, `embedding`).
- Creates the `match_documents` PostgreSQL function used to perform cosine similarity searches.

## Setup Instructions

1. **Install Dependencies**
   Ensure you have configured a virtual environment and then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory with the following variables (if using Supabase/HuggingFace APIs):
   ```
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   HF_TOKEN=your_huggingface_api_token
   ```

3. **Database Setup (Supabase)**
   If using the Supabase scripts (`1C_...` or `4A_...`), execute the contents of `schema.sql` in your Supabase SQL Editor.

4. **Add Documents**
   Create a `docs/` folder in the root directory and place your `.txt` files inside.

5. **Run Ingestion**
   Execute one of the ingestion scripts to build the vector store:
   ```bash
   python Step1_ingestion_semantic_chunking.py
   ```

6. **Start the Chat Interface**
   Run the generation script to query your documents:
   ```bash
   python Step2_history_aware_generation_supabase.py
   ```
