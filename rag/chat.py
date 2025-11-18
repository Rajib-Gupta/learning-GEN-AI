from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
load_dotenv()  # Load environment variables from a .env file

client = OpenAI()
# Langchain openAi embedding model to embed the chunks
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

#Connect to the existing Qdrant vector store
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="nodejs_docs"
)



# Take User Query
USER_QUERY = input("Enter your query: ")

# Perform similarity search on the vector database to retrieve relevant chunks
search_results = vector_db.similarity_search(
    query=USER_QUERY
)

# Combine the retrieved chunks to form context for answering the query
context = "\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\n File Location: {result.metadata['source']}" for result in search_results])

SYSTEM_PROMPT = f"""You are a helpful AI assistant who answers user query based on the available context retreive from a PDF file along with the page_content and page number.
If the context does not contain the answer, respond with "I don't know". 
You should only answer to the user based on the following context and redirect to user that page to know more about the context.
Context: {context}


"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": USER_QUERY
        }
    ]
)

print("AI Assistant Response:" , response.choices[0].message.content)