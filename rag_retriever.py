from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Configuration ---
# Example Terraform URLs (Expand this list for real docs!)
TERRAFORM_DOCS_URLS = [
    "https://developer.hashicorp.com/terraform/language/syntax/configuration",
    "https://developer.hashicorp.com/terraform/language/resources",
    # Add an AWS provider resource page for testing the S3 example
    "https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket"
]
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Your chosen Ollama embedding model

class RAGRetriever:
    def __init__(self):
        print(f"--- RAG SETUP: Loading docs from {len(TERRAFORM_DOCS_URLS)} URLs ---")
        
        # 1. Load Documents using WebBaseLoader
        # WebBaseLoader is suitable for scraping standard HTML pages like docs
        loader = WebBaseLoader(TERRAFORM_DOCS_URLS)
        # Load and Split documents
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=400,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        print(f"--- RAG SETUP: Split into {len(splits)} chunks ---")
        
        # 2. Embed and Store (Using OllamaEmbeddings)
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
        
        # Chroma is used for simplicity, but can be replaced with a persistent store later
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        print("--- RAG SETUP COMPLETE ---")

    def get_context(self, query: str) -> str:
        """Retrieves and formats relevant context for the LLM."""
        retrieved_docs = self.retriever.invoke(query)
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        return context

# Instantiate the retriever for use in the agent
iac_retriever = RAGRetriever()
