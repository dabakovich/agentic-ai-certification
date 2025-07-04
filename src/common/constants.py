from langchain.prompts import PromptTemplate

# User-focused prompt with documents context
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Based on the following publications, answer the users's question:

Documents context:
{context}

User's question: {question}

Answer: Provide a comprehensive answer based on the documents above.
""",
)
