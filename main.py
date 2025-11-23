from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# model
model= OllamaLLM(model="gemma3:1b", stream=True)

template = """
you are an expert in answering questions about a pizza restuarant 
here are some reviews {reviews}

Here is the question to answer {question}
"""

prompt = ChatPromptTemplate.from_template(template)

#connect prompt to model
chain = prompt | model 

# Control flow of application
while True: 
    print("\n\n---------------------------------")
    question = input("Ask a question: ")
    print('\n\n')
    if question=="q":
        break

    reviews = retriever.invoke(question)    
    # result = chain.invoke({"reviews":reviews, "question":question})
    
    for chunk in chain.stream({"reviews":reviews, "question":question}): 
        print(chunk, end='', flush=True)
    
    