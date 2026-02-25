from langsmith import traceable
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from custom_ollama import ChatOllamaWithUsage
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
from transformers import AutoTokenizer
from langsmith import trace

# Load tokenizer once globally
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "docs.txt"

# --------------------------------------------------
# Load documents ONLY from docs.txt
# --------------------------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    # Split by blank lines to keep requirement sections together
    documents = [
        chunk.strip()
        for chunk in f.read().split("\n\n")
        if chunk.strip()
    ]

# --------------------------------------------------
# Create embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
doc_embeddings = embeddings.embed_documents(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# --------------------------------------------------
# Store embeddings in FAISS (vector DB)
# --------------------------------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# --------------------------------------------------
# Retriever
# --------------------------------------------------
#def retrieve(query, top_k=4):
    #query_embedding = embeddings.embed_documents([query])
    #distances, indices = index.search(np.array(query_embedding), top_k)
    #return [documents[i] for i in indices[0]]

def retrieve(query, embeddings, index, documents, top_k=4):
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# --------------------------------------------------
# Initialize LLM (Ollama - Llama3)
# --------------------------------------------------
llm = ChatOllamaWithUsage(model="llama3", temperature=0)


# --------------------------------------------------
# Extract generation stats
# --------------------------------------------------
def extract_generation_stats(ai_message):

    usage = getattr(ai_message, "usage_metadata", {}) or {}
    meta = getattr(ai_message, "response_metadata", {}) or {}

    # Primary source (LangChain standard)
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    # Fallback for Ollama-style metadata
    if prompt_tokens is None:
        prompt_tokens = meta.get("prompt_eval_count")

    if completion_tokens is None:
        completion_tokens = meta.get("eval_count")

    # Final fallback
    if total_tokens is None:
        if prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens
        else:
            total_tokens = None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency": meta.get("total_duration"),
    }


# --------------------------------------------------
# Generator
# --------------------------------------------------
def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a QA engineer.

Using ONLY the requirements provided below, generate clear and structured test cases.
Each test case should include:
- Test Case ID
- Description
- Preconditions
- Steps
- Expected Result

Requirements:
{context_text}

Task:
{query}
"""

    ai_message = llm.invoke(prompt)

    stats = extract_generation_stats(ai_message)

    with trace(
        name="llama3_generation",
        metadata={
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "total_tokens": stats["total_tokens"],
            "latency": stats["latency"],
        },
    ):
        pass  # metadata will attach to this trace block

    return ai_message.content, stats


# --------------------------------------------------
# RAG entry point
# --------------------------------------------------
@traceable(name="rag_pipeline")
def rag_answer(query):
    contexts = retrieve(query, embeddings, index, documents)
    answer, stats = generate_answer(query, contexts)
    return answer, contexts, stats

# --------------------------------------------------
# Generator (Ollama)
# --------------------------------------------------
#def generate_answer(self, query, contexts):
  #  context_text = "\n\n".join(contexts)

    #prompt = f"""
#You are a QA engineer.

#Using ONLY the requirements provided below, generate clear and structured test cases.
#Each test case should include:
#- Test Case ID
#- Description
#- Preconditions
#- Steps
#- Expected Result

#Requirements:
#{context_text}

#Task:
#{query}
# """
    
#     ai_message = self.llm.invoke(prompt)

#     # Extract token + generation stats
#     generation_stats = self._extract_generation_stats(ai_message)

#     return ai_message.content, generation_stats

#     def _extract_generation_stats(self, ai_message):

#         usage = getattr(ai_message, "usage_metadata", {}) or {}
#         meta = getattr(ai_message, "response_metadata", {}) or {}

#     # Primary source (LangChain standard)
#     prompt_tokens = usage.get("input_tokens")

#     completion_tokens = usage.get("output_tokens")

#     total_tokens = usage.get("total_tokens")

#     # Fallback for Ollama-style metadata
#     if prompt_tokens is None:
#         prompt_tokens = meta.get("prompt_eval_count")

#     if completion_tokens is None:
#         completion_tokens = meta.get("eval_count")

#     # Final fallback
#     if total_tokens is None:
#         if prompt_tokens and completion_tokens:
#             total_tokens = prompt_tokens + completion_tokens
#         else:
#             total_tokens = None

#     return {
#         "prompt_tokens": prompt_tokens,
#         "completion_tokens": completion_tokens,
#         "total_tokens": total_tokens,
#         "latency": meta.get("total_duration"),
#     }


    # from langsmith import trace

    #with trace(
       # name="llama3-generation",
        #metadata={
         #   "prompt_tokens": prompt_tokens,
          #  "completion_tokens": completion_tokens,
           # "total_tokens": total_tokens,
       # },#
    #):#
       # pass  # metadata will attach to this trace block

    #return answer_text */
# --------------------------------------------------
# RAG entry point
# --------------------------------------------------
#def rag_answer(query):
    #contexts = retrieve(query)
    #answer = generate_answer(query, contexts)
    #return answer, contexts

#def rag_answer(query):
    #contexts = retrieve(query, embeddings, index, documents)
    #answer = generate_answer(query, contexts)
    #return answer, contexts

    
    #contexts = retrieve(query, embeddings, index, documents)
    #answer, stats = generate_answer(query, contexts)
    #return answer, contexts, stats


# --------------------------------------------------
# Command-line usage
# --------------------------------------------------
if __name__ == "__main__":
    question = input("\nEnter your question: ")

    answer, contexts, stats = rag_answer(question)

    print("\n================ Retrieved Contexts ================")
    for i, ctx in enumerate(contexts, 1):
        print(f"{i}. {ctx}")

    print("\n================ Generated Answer =================")
    print(answer)
