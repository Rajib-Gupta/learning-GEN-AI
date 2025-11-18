from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()  # Load environment variables from a .env file

# Define the path to the PDF file
file_path = Path(__file__).parent / "nodejs.pdf"

# Loads the PDF document with the help of langchain community loaders
loader = PyPDFLoader(file_path = str(file_path))

doc = loader.load()  # Load the document page by page
print(f"Loaded {len(doc)} pages from the document.")
print(doc[0])  # Print the content of the first page

# Split the document into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, # Size of each chunk
    chunk_overlap = 400, # Overlap between chunks to maintain context
)

chunks = text_splitter.split_documents(documents=doc)  # Split the document into smaller chunks

print("first chunk",chunks[0])  # Print the content of the first chunk

# Langchain openAi embedding model to embed the chunks

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

# Store the embedding Chunks in a vector database like Qdrant for retrieval
vectorStore = QdrantVectorStore.from_documents(
    documents= chunks,
    embedding= embedding_model,
    url="http://localhost:6333",
    collection_name="nodejs_docs"
)

print("Vector Store Created Successfully")