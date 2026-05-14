from pipelines.rag_pipeline import run



if __name__ == "__main__":
    
    print("Type 'exit' to quit.\n")

    while True:
        query_text = input("Question: ").strip()
        if not query_text or query_text.lower() == "exit":
            break

        answer = run(query_text)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)
