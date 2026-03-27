import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# Load environment variables
load_dotenv()

# Connect to Supabase Document Database
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env file.")

supabase: Client = create_client(supabase_url, supabase_key)

# Initialize embeddings matching what was used during ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="TaylorAI/bge-micro-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

from typing import Any, List, Tuple
from langchain_core.documents import Document

class PatchedSupabaseVectorStore(SupabaseVectorStore):
    """
    Workaround for LangChain's SupabaseVectorStore which breaks with newer 
    versions of the `supabase` / `postgrest-py` pip packages because 
    SyncRPCFilterRequestBuilder no longer has a .params attribute.
    """
    def similarity_search_by_vector_with_relevance_scores(
        self, query: List[float], k: int = 4, filter: dict[str, Any] | None = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        vectors = query
        match_documents_params = dict(
            query_embedding=vectors, match_count=k
        )
        if filter:
            match_documents_params["filter"] = filter

        res = self._client.rpc(self.query_name, match_documents_params).limit(k).execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}), 
                    page_content=search.get("content", "")
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]
        return match_result

# Initialize the vector store pointing to our Supabase table
db = PatchedSupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Set up AI model for generation locally to avoid API Gateway Timeouts
# We use SmolLM-135M locally to minimize token size and RAM usage.
llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceTB/SmolLM-135M-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.3,
        "do_sample": True
    },
)

model = ChatHuggingFace(llm=llm)

# Store our conversation as messages
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")
    
    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]
        
        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question
    
    # Step 2: Find relevant documents from Supabase
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)
    
    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print("\n" + "-" * 50)
    print("ANSWER")
    print("-" * 50)
    print(f"{answer}\n")
    return answer

# Simple chat loop
def start_chat():
    print("Welcome to the Supabase-backed History-Aware RAG Assistant!")
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        try:
            question = input("\nYour question: ")
            
            if question.lower() == 'quit':
                print("Goodbye!")
                break
                
            ask_question(question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    start_chat()
