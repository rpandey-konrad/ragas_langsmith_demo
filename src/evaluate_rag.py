import os
from dotenv import load_dotenv

# Load secrets from environment/.env; keep credentials out of the repo.
load_dotenv()
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "ragas-demo")
os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "false")
from custom_ollama import ChatOllamaWithUsage
#from langchain_community.chat_models import ChatOllama # ---- Ollama LLM ----
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,
)
from datasets import Dataset
from langsmith import trace
from rag_pipeline import rag_answer


# ---- RAGAS wrappers ----
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langsmith import traceable



# ---- Local embeddings ----
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings



def main():
    # -------------------------
    # Configure LOCAL LLM (Ollama)
    # -------------------------
    ollama_llm = ChatOllamaWithUsage(
        model="llama3",
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(ollama_llm)

    # -------------------------
    # Configure LOCAL embeddings
    # -------------------------
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # -------------------------
    # -------------------------
    # Dynamic question (terminal input)
    # -------------------------
    question = input("\nEnter a question for RAG evaluation: ")
    answer, contexts, stats = rag_answer(question)



    # -------------------------
    # 🔍 Print RAG output (demo-friendly)
    # -------------------------
    print("\n================ RAG OUTPUT ================")
    print("Question:")
    print(question)

    print("\nRetrieved Contexts:")
    for i, ctx in enumerate(contexts, 1):
        print(f"{i}. {ctx}")

    print("\nGenerated Answer:")
    print(answer)
    print("===========================================\n")

    # -------------------------
    # Prepare dataset
    # -------------------------
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        
    }

    dataset = Dataset.from_dict(data)

    # -------------------------
    # Run RAGAS evaluation
    # -------------------------
    with trace(name="ragas_evaluation"):
        results = evaluate(
        dataset,
        metrics=[
            faithfulness,
           answer_relevancy,
            context_utilization,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,  
    )
   
    print("\nRAGAS Evaluation Results:")
    print(results)


if __name__ == "__main__":
    main()
