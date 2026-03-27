import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from dotenv import load_dotenv

# Note: Before running this script, you MUST:
# 1. Have a Supabase project created (https://supabase.com/)
# 2. Add SUPABASE_URL and SUPABASE_SERVICE_KEY to your .env file
# 3. Enable the pgvector extension and create the `documents` table in Supabase SQL editor:
"""
-- Run this in your Supabase SQL editor:
create extension if not exists vector;

create table documents (
  id uuid primary key default uuid_generate_v4(),
  content text, -- corresponds to page_content
  metadata jsonb, -- corresponds to metadata
  embedding vector(384) -- 384 is the dimension for all-MiniLM-L6-v2
);

create function match_documents (
  query_embedding vector(384),
  match_count int DEFAULT null,
  filter jsonb DEFAULT '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;
"""

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    for i, doc in enumerate(documents): 
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def semantic_split_documents(documents):
    """Split documents using SemanticChunker to group by meaning"""
    print("\nSplitting documents into semantic chunks...")
    
    # Initialize the Semantic Chunker with sentence-transformers embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="TaylorAI/bge-micro-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",  
        breakpoint_threshold_amount=50 # Decreased to 50 to make smaller chunks and use fewer tokens
    )
    
    # split_documents automatically handles returning Document objects with metadata preserved
    chunks = semantic_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Semantic Chunk {i+1} ---")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content preview:")
            print(chunk.page_content[:200] + "...")
            print("-" * 50)
        
        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more semantic chunks.\n")
            
    return chunks

def create_vector_store(chunks):
    """Create and persist Supabase vector store"""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env file.")
        
    print(f"Connecting to Supabase at {supabase_url}...")
    supabase: Client = create_client(supabase_url, supabase_key)
        
    embedding_model = HuggingFaceEmbeddings(
        model_name="TaylorAI/bge-micro-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create Supabase vector store
    print("--- Storing vectors in Supabase ---")
    print("This may take some time depending on your network connection...")
    
    vectorstore = SupabaseVectorStore.from_documents(
        chunks,
        embedding_model,
        client=supabase,
        table_name="documents",
        query_name="match_documents"
    )
    print("--- Finished storing vectors in Supabase ---")
    
    return vectorstore

def main():
    """Main ingestion pipeline using Semantic Chunking and Supabase"""
    print("=== RAG Semantic Document Ingestion Pipeline (Supabase Edition) ===\n")
    
    docs_path = "docs"
    
    # Step 1: Load documents
    documents = load_documents(docs_path)  

    # Step 2: Split into Semantic chunks
    chunks = semantic_split_documents(documents)
    
    # Step 3: Create vector store (Upload to Postgres / Supabase)
    vectorstore = create_vector_store(chunks)
    
    print("\n✅ Supabase Ingestion complete! Your documents are now in the cloud and ready for RAG queries.")

if __name__ == "__main__":
    main()
