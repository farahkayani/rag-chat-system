# rag-chat-system
A custom Retrieval-Augmented Generation (RAG) chat system with a local retriever and domain-specific dataset.
# RAG Chat System

This project implements a **Retrieval-Augmented Generation (RAG)** system using FAISS and a local dataset to simulate intelligent, context-aware response generation. The system is designed to retrieve the most relevant documents based on a user query and generate an informed answer using a prompt-based language model. It is modular, consisting of separate notebooks for indexing, querying, generation, and evaluation.

---

## Project Structure

RAG chat system/
├── dataset1.csv # Knowledge base used for document retrieval
├── retriever_index.faiss # Pre-built FAISS vector index
├── retriever.ipynb # Code for indexing and retrieving relevant documents
├── main.ipynb # Executes retrieval + generation workflow
├── evaluate.ipynb # Evaluates response quality using metrics
├── prompt.txt # Template prompt used to guide the LLM
├── generation_output.txt # Stores generated output from the model
├── Report.pdf # Complete documentation of the system
└── README.md # Project documentation (this file)


---

##  Features

- **Document Retrieval:** Utilizes FAISS to find relevant documents from a local dataset.
- **Prompted Response Generation:** Integrates a customizable prompt system to guide the language model output.
- **Evaluation System:** Includes performance metrics and manual inspection tools to assess system effectiveness.
- **Modular Notebooks:** Clean separation of retriever, generator, and evaluator components.
- **Dataset-based QA:** Responses are tightly coupled with the dataset content, simulating real-world RAG systems.

---

## Dataset

- **File:** `dataset1.csv`
- **Purpose:** Serves as the source knowledge base for the retrieval system.
- **Content:** Each row represents a document, passage, or fact used to generate responses.

---

## 🛠️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/farahkayani/rag-chat-system
cd rag-chat-system
