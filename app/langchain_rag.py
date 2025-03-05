import os
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

os.environ['OPENAI_API_KEY'] = '{{OPEN_AI}}'

documents = [
    "Python is a high-level programming language known for its readability and versatile libraries.",
    "Java is a popular programming language used for building enterprise-scale applications.",
    "JavaScript is essential for web development, enabling interactive web pages.",
    "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
    "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
    "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
    "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
    "At Educative, we think RAG is the future of AI!",
]
# Create a Chroma database from the documents using OpenAI embeddings
db = Chroma.from_texts(documents, OpenAIEmbeddings())
print("db initialized")

# Configure the database to act as a retriever, setting the search type to similarity and returning the top 1 result
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)

# Define a template for generating answers using provided context
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""

# Create a custom prompt template using the defined template
custom_rag_prompt = PromptTemplate.from_template(template)
print(custom_rag_prompt)  # Print the custom prompt template

# Perform a similarity search with the given query
question = "What is the future of AI?"
llm = ChatOpenAI(model="gpt-4o-mini")  # Initialize the language model with the specified model
rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # Pass the context and question
        | custom_rag_prompt  # Format the prompt using the custom RAG prompt template
        | llm  # Use the language model to generate a response
        | StrOutputParser()  # Parse the output to a string
)

# Invoke the RAG chain with a question
response = rag_chain.invoke(question)
print(response)  # Print the response
