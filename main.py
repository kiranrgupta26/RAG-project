from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import ollama
from pypdf import PdfReader
import chromadb
from datetime import datetime

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_chunks")
    
def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text, reader.pages
    
     
def get_metadata(chunks, file_name):

    metadata = []
    extraction_date = datetime.now().strftime("%Y-%m-%d")

    for i in range(len(chunks)):
        metadata.append({
            "file_name": file_name,
            "chunk_index": i,
            "date": extraction_date
        })

    return metadata
    
def get_chunks(documents):
    chunks = splitter.split_text(documents[0])
    return chunks
    
 
def get_embedding_vectors(chunks):
    embeddings = embed_model.encode(chunks)
    
    return embeddings
  
def add_to_vector_db(chunks,embeddings,metadata):
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))],
        metadatas=metadata
    )
    
def retrieve(query, k=3):
    q_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=k
    )
    docs = results.get("documents", [[]])[0]
    meta = results.get("metadatas", [[]])[0]
    
    return docs,meta


def ask_llm(question):
    context_chunks,metadata = retrieve(question)
    
    print(f"Metadata is: {metadata}")
    if not context_chunks:
        return "Not found in the provided context."
    
    context = "\n\n".join(context_chunks)
    
    prompt = f"""You are a strict question-answering system.

RULES:
- Use the Context as the primary source of truth.
- You may **infer or paraphrase** the answer from the context if necessary.
- Do NOT use any knowledge outside the Context.
- If the answer cannot be reasonably inferred from the Context, reply exactly:
  "Not found in the provided context."

<CONTEXT>
{context}
</CONTEXT>

Question:
{question}

Answer (verbatim from the Context):
"""
    
    response = ollama.chat(
        model="mistral",
        messages=[{"role":"user","content":prompt}]
    )
    
    return response["message"]["content"]
   

    
file_name = "bank.pdf"    
text, pages = read_pdf("bank.pdf")
documents = [text]

chunks = get_chunks(documents)

metadata = get_metadata(chunks, file_name)

embeddings = get_embedding_vectors(chunks)

add_to_vector_db(chunks,embeddings,metadata)

while True:
    i_question = input("Ask Question: ")
    result = ask_llm(i_question)
    print(f"Result: {result}")
