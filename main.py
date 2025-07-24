from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver

model = OllamaLLM(model="llama3.2")

template = """
Your are an expert in answering questions about a pizza restaurant.
Here are some relevant review: {reviews}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("\n\n-----------------------------------")
    question = input("Enter your question (or 'q' to quit): ")
    if question.lower() == 'q':
        break
    
    reviews = retriver.invoke(question)

    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)

