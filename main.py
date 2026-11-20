import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open ("data/knowledge.json", "r",encoding="utf-8") as file:
    documents= json.load(file)


texts=[doc["content"]for doc in documents]

vectorizer=TfidfVectorizer()

document_vectors= vectorizer.fit_transform(texts)

def retrieve(question):
    question_vector=vectorizer.transform([question])

    similarities=cosine_similarity(
     question_vector, document_vectors
     )[0]
    best_index=similarities.argmax()
    return documents[best_index]["content"]

question=input("Ask your question :")
result=retrieve(question)
print("\nRetrieved Document:")
print(result)