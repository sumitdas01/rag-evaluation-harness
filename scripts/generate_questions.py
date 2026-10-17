import json
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.config import config
from src.retrieval.vector_retriever import VectorRetriever

def generate_questions():
    """
    Helper script to generate a ground truth evaluation dataset from existing documents.
    Note: Requires the vector index to be built already.
    """
    output_file = Path(config.paths.evaluation_questions)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have questions
    if output_file.exists():
        print(f"Questions file already exists at {output_file}. Skipping generation.")
        return

    # Initialize LLM
    try:
        llm = ChatOpenAI(model=config.llm.model, temperature=config.llm.temperature)
    except Exception as e:
        print(f"Error initializing LLM: {e}. Make sure OPENAI_API_KEY is set.")
        return

    # Use a vector retriever just to get a sample of chunks to generate questions from.
    # We assume 'simple' strategy index exists.
    try:
        retriever = VectorRetriever(strategy="simple", top_k=1)
        docs = retriever.get_all_documents()
    except Exception as e:
        print(f"Failed to get documents: {e}. Please build the simple index first.")
        return

    if not docs:
        print("No documents found in the simple index.")
        return

    prompt = PromptTemplate.from_template("""
    You are an AI assistant tasked with generating evaluation questions for a RAG system.
    Based on the following document chunk, generate ONE question that can be explicitly answered by this chunk.
    Also provide the expected answer.
    
    Document Chunk:
    {context}
    
    Return the result exactly as a JSON object with two keys: "question" and "expected_answer".
    Do not include markdown blocks like ```json ... ```, just output the raw JSON.
    """)

    questions = []
    
    # Generate up to 50 questions
    num_questions = min(50, len(docs))
    print(f"Generating {num_questions} questions...")
    
    for i in range(num_questions):
        doc = docs[i]
        try:
            chain = prompt | llm
            response = chain.invoke({"context": doc.page_content})
            result = json.loads(response.content.strip())
            
            question_data = {
                "id": f"Q{str(i+1).zfill(3)}",
                "question": result["question"],
                "expected_answer": result["expected_answer"],
                "relevant_chunk_ids": [doc.metadata.get("chunk_id", "")],
                "source_documents": [doc.metadata.get("source", "")]
            }
            questions.append(question_data)
            print(f"Generated Q{i+1}: {result['question']}")
        except Exception as e:
            print(f"Failed to generate question {i+1}: {e}")

    with open(output_file, "w") as f:
        json.dump(questions, f, indent=4)
        
    print(f"\nGenerated {len(questions)} questions and saved to {output_file}")
    print("IMPORTANT: Please manually review and edit this file to ensure quality!")

if __name__ == "__main__":
    generate_questions()
