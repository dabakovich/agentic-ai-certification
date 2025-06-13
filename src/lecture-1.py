from utils import get_llm, load_env, load_publication
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_env()

publication_content = load_publication()

llm = get_llm()

# Initialize conversation
conversation = [
    SystemMessage(content=f"""
You are a helpful AI assistant discussing a research publication.
Base your answers only on this publication content:

{publication_content}
""")
]

# User question 1
conversation.append(HumanMessage(content="""
What are variational autoencoders and list the top 5 applications for them as discussed in this publication.
"""))

response1 = llm.invoke(conversation)
print("🤖 AI Response to Question 1:")
print(response1.content)
print("\n" + "="*50 + "\n")

# Add AI's response to conversation history
conversation.append(AIMessage(content=response1.content))

# User question 2 (follow-up)
conversation.append(HumanMessage(content="""
How does it work in case of anomaly detection?
"""))

response2 = llm.invoke(conversation)
print("🤖 AI Response to Question 2:")
print(response2.content)