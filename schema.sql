-- Enable the pgvector extension to work with embeddings
create extension if not exists vector;

-- Create a table to store your documents
create table documents (
  id uuid primary key default uuid_generate_v4(),
  content text, 
  metadata jsonb, 
  embedding vector(384) -- 384 is the dimension size for sentence-transformers/all-MiniLM-L6-v2
);

-- Create a function to search for documents
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
