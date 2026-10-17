import os
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.config import config

class AnswerGenerator:
    def __init__(self):
        self.llm = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=config.llm.model, 
                    temperature=config.llm.temperature
                )
            except Exception as e:
                print(f"Warning: Could not initialize ChatOpenAI: {e}")

        self.prompt = PromptTemplate.from_template("""
        You are a helpful assistant answering questions based on the provided context.
        If the answer is not in the context, say "I cannot answer this based on the provided context."
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """)
        
    def generate(self, question: str, retrieved_docs: List[Document]) -> str:
        """Generates an answer based on retrieved documents."""
        if not retrieved_docs:
            return "No context retrieved."
            
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        if self.llm:
            try:
                chain = self.prompt | self.llm
                response = chain.invoke({
                    "context": context,
                    "question": question
                })
                return response.content.strip()
            except Exception as e:
                pass  # Fallback to extractive answer below
                
        # Extractive fallback when LLM is unavailable:
        # Extract sentences from context that match question terms
        import re
        q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())) - {"what", "who", "when", "which", "how", "under", "with", "from", "that", "this"}
        sentences = re.split(r'(?<=[.!?\n])\s+', context)
        scored_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 20:
                s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s_clean.lower()))
                overlap = len(q_words.intersection(s_words))
                scored_sentences.append((overlap, s_clean))
                
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for score, s in scored_sentences[:3] if score > 0]
        if top_sentences:
            return " ".join(top_sentences)
        return context[:300] + "..."
