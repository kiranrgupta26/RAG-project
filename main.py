from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import ollama
from pypdf import PdfReader
import chromadb
from datetime import datetime

def store_metadata(chunks, file_name, pages):
    metadata = []
    # Extracting date dynamically (you could use a fixed date if needed)
    extraction_date = datetime.now().strftime("%Y-%m-%d")
    
    for page_number, page in enumerate(pages):
        for i, chunk in enumerate(chunks):
            metadata.append({
                "file_name": file_name,
                "page_number": page_number + 1,
                "date": extraction_date,
                "chunk_index": i
            })
    
    return metadata
    
def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text, reader.pages
    
file_name = "bank.pdf"    
text, pages = read_pdf("bank.pdf")

documents = [text]
splitter = RecursiveCharacterTextSplitter(chunk_size = 200,
                                         chunk_overlap = 50)

chunks = splitter.split_text(documents[0])

metadata = store_metadata(chunks, file_name, pages)
#Create Embeddings

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks)

# Chroma Database
client = chromadb.PersistentClient(path="./chroma_db")

#Create Collection
collection = client.get_or_create_collection(name="pdf_chunks")

# Add embeddings and documents to Chroma
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),  # Chroma expects a list of lists
    ids=[str(i) for i in range(len(chunks))],
    metadatas=metadata
)

#Retireve Function
def retrieve(query, k=3):
    q_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=k
    )
    docs = results.get("documents", [[]])[0]
    meta = results.get("metadatas", [[]])[0]
    
    return docs,meta
    
#Connect to Local LLM and ask question

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


while True:
    i_question = input("Ask Question: ")
    result = ask_llm(i_question)
    print(f"Result: {result}")
