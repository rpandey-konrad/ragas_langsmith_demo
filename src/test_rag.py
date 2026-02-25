from rag_pipeline import rag_answer


question = "What is RAGAS?"

# rag_answer returns answer, contexts, and stats (token/latency info)
answer, contexts, stats = rag_answer(question)

print("\nQUESTION:")
print(question)

print("\nRETRIEVED CONTEXTS:")
for c in contexts:
    print("-", c)

print("\nMODEL ANSWER:")
print(answer)

print("\nGENERATION STATS:")
print(stats)
