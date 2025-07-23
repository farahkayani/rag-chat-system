import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np

df = pd.read_csv("C:/Users/farah/OneDrive/Desktop/AI SUMMER ASSIGNMENT1/dataset1.csv", encoding="utf-8")
print("Dataset loaded")
print(df.head())


documents = df['answer'].tolist()

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()


dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_vectors, dtype='float32'))


query = " how big is a panda?"
query_vec = vectorizer.transform([query]).toarray().astype('float32')

D, I = index.search(query_vec, k=3)


retrieved_docs = [documents[i] for i in I[0]]
print("Top 3 Retrieved Answers:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i}. {doc}")


prompt = f"""Context:
1. {retrieved_docs[0]}
2. {retrieved_docs[1]}
3. {retrieved_docs[2]}

Question:
{query}

Answer:"""

with open("C:/Users/farah/OneDrive/Desktop/AI SUMMER ASSIGNMENT1/prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

print("\n Prompt saved to 'prompt.txt' ")
