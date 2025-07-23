import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np

#loading dataset
df = pd.read_csv("C:/Users/farah/OneDrive/Desktop/AI SUMMER ASSIGNMENT1/dataset1.csv")
print("Dataset loaded")
print(df.head())

#Preprocess Answers
documents = df['answer'].tolist()

#Convert to Vectors Using TF-IDF
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents).toarray()

#Index with FAISS as FAISS enables fast similarity search over vectors.
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)

# Convert to float32 for FAISS
index.add(np.array(doc_vectors, dtype='float32'))

# Save index
faiss.write_index(index, "retriever_index.faiss")

#Retrieval Example
query = "What are flu symptoms?"
query_vec = vectorizer.transform([query]).toarray().astype('float32')

# Search top 3 relevant answers
D, I = index.search(query_vec, k=3)
print("Top Matches:")
for idx in I[0]:
    print(f"- {documents[idx]}")
