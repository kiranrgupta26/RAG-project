from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import ollama
from pypdf import PdfReader
import chromadb


def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
    
    
text = read_pdf("bank.pdf")
documents = [text]
splitter = RecursiveCharacterTextSplitter(chunk_size = 200,
                                         chunk_overlap = 50)

chunks = splitter.split_text(documents[0])


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
    ids=[str(i) for i in range(len(chunks))]
)

#Retireve Function
def retrieve(query, k=3):
    q_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=k
    )
    docs = results.get("documents",[[]])[0]
    return docs
    
#Connect to Local LLM and ask question

def ask_llm(question):
    context_chunks = retrieve(question)
   
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
