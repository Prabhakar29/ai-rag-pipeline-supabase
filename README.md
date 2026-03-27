# ai-rag-pipeline-supabase

This project implements a Retrieval-Augmented Generation (RAG) pipeline leveraging Langchain, HuggingFace, and Supabase. The configuration is optimized for minimal footprint models that can run efficiently locally.

## Architecture Highlights
- **Document Loading:** Reads `.txt` files from a local `docs/` directory.
- **Embeddings:** Uses local HuggingFace embedding model (`TaylorAI/bge-micro-v2`) via `HuggingFaceEmbeddings`.
- **LLM:** Uses local inference with `HuggingFaceTB/SmolLM-135M-Instruct` via `HuggingFacePipeline` for response generation.
- **Vector Stores:** Integrates with cloud-based Supabase using the `pgvector` extension.
- **Chunking Strategies:** Incorporates advanced **semantic chunking** to intelligently group sentences.
- **History-Aware Generation:** Includes a terminal chat interface that remembers conversation history and rephrases questions contextually.

## Key Files

### `Step1_ingestion_semantic_chunking.py`
The ingestion script responsible for data preparation:
1. Loads documents from the `docs/` folder.
2. Uses Langchain's `SemanticChunker` to semantically group sentences together with an optimized threshold.
3. Retrieves or generates vector embeddings using `TaylorAI/bge-micro-v2`.
4. Stores these embeddings directly into your **Supabase** database. Requires 'HF_TOKEN' `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in your `.env`.

### `Step2_history_aware_generation_supabase.py`
The interaction script that interfaces with your vector store:
1. Connects to the Supabase vector store populated earlier.
2. Initializes the minimal RAG generation language model (`HuggingFaceTB/SmolLM-135M-Instruct`) locally.
3. Provides a continuous conversational terminal loop (`start_chat`).
4. Re-phrases follow-up queries based on conversational history.
5. Retreives relevant context and synthesizes natural language answers.
*Note: Contains a patch for LangChain's Supabase vector store integration to properly handle filtering with newer Supabase client versions.*

### `schema.sql`
A SQL script containing the expected database schema for Supabase:
- Enables the pgvector extension.
- Creates the `documents` table (`id`, `content`, `metadata`, `embedding`). *Note: Dimension size is 384 for `bge-micro-v2`.*
- Creates the `match_documents` PostgreSQL function used to perform cosine similarity searches.

## Setup Instructions

1. **Install Dependencies**
   Ensure you have configured a virtual environment and then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory with your Supabase credentials.
   ```
   HF_TOKEN=your_huggingface_api_token
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   ```
   *(If you wish to use HuggingFace inference APIs instead of running locally, you also need to set `HF_TOKEN`)*

3. **Database Setup (Supabase)**
   Execute the contents of `schema.sql` in your Supabase SQL Editor to set up the `pgvector` table and similarity search function.

4. **Add Documents**
   Create a `docs/` folder in the root directory and place your data or `.txt` files inside.

5. **Run Ingestion**
   Execute the ingestion script to process your documents and build the vector store within Supabase:
   ```bash
   python Step1_ingestion_semantic_chunking.py
   ```

6. **Start the Chat Interface**
   Once data is successfully ingested, start the generation script to query your documents:
   ```bash
   python Step2_history_aware_generation_supabase.py
   ```
