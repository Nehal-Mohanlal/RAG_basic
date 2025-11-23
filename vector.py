from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os 
import pandas as pd 

## read the data 
df = pd.read_csv("realistic_restaurant_reviews.csv")

#embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# where we create our embedding db
db_location = "./chroma_langchain_db"

add_documents = not os.path.exists(db_location)

## retrieve data 
if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows(): 
            document = Document(
                page_content=row["Title"] + " " + row["Review"], 
                metadata= {"rating": row["Rating"], "date":row["Date"]}, 
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)
            

## add documents to db 
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function= embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
## what we call in main.py
retriever = vector_store.as_retriever(
    search_kwargs= {"k":5}
)