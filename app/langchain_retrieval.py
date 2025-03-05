import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

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
]
# Create a Chroma database from the documents using OpenAI embeddings
db = Chroma.from_texts(documents, OpenAIEmbeddings())

# Configure the database to act as a retriever, setting the search type to
# similarity and returning the top 1 result
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)

# Perform a similarity search with the given query
result = retriever.invoke("Where can I see Mona Lisa?")
print(result)
