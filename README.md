# Gemma Model Document Q&A using Streamlit, LangChain, and FAISS

Demo video link :- https://drive.google.com/file/d/1NPoG--zozneJ8b4b01QYsZWITrB_i27D/view?usp=sharing

This project is a document-based Question & Answer (Q&A) system built using the Gemma model (via Groq), LangChain, Google Generative AI embeddings, FAISS vector store, and Streamlit. The application allows users to load PDFs, create embeddings, and ask natural language questions based on document content.

---

## Features

- Load and process PDF documents from a directory
- Split documents into manageable text chunks
- Generate semantic vector embeddings using Google's embedding model
- Store vectors in a local FAISS vector store
- Retrieve relevant document chunks based on the user’s question
- Use the Gemma model to answer questions using only the document content
- Simple and interactive Streamlit interface

---

## Technologies Used

- **Frontend/UI**: Streamlit
- **Large Language Model**: Gemma 2-9B (via Groq)
- **Embeddings**: Google Generative AI Embedding model (embedding-001)
- **Vector Database**: FAISS
- **Document Parsing**: LangChain's PyPDFDirectoryLoader
- **Framework**: LangChain

---

## Folder Structure

```
project-root/
│
├── us_census/               # Folder containing PDF files
├── .env                     # Environment variables file
├── main.py                  # Streamlit app script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Environment Variables

Create a `.env` file in the project directory and add:

```
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

## How to Run the Project

### 1. Clone the repository

### 2. Create and activate a virtual environment (optional but recommended)

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add PDF files

Place all your documents in the `us_census/` folder. This is the default folder used by the app to load documents.

### 5. Run the Streamlit app

```bash
streamlit run main.py
```

---

## How It Works

1. **Embedding Step**
   - PDF files are loaded from the specified folder.
   - The content is split into smaller chunks using `RecursiveCharacterTextSplitter`.
   - These chunks are converted into vector embeddings using Google's embedding model.

2. **Vector Store Creation**
   - All embeddings are stored in an in-memory FAISS vector store.
   - This step is triggered manually through the Streamlit UI using a "Create vector store" button.

3. **Question Answering**
   - User enters a question.
   - The system uses the FAISS retriever to fetch the most relevant document chunks.
   - These chunks are passed to the Gemma model to generate an answer.
   - The model strictly answers based on the retrieved context and avoids hallucinations.

---

## Example Usage

- Start the app and click on “Create vector store” to build the vector database from PDFs.
- Enter questions like:
  - "What is the population growth rate?"
  - "Who conducted the 2020 census?"
- The app will respond with detailed answers based only on the document content.

---

## Dependencies

The project requires the following libraries (all listed in `requirements.txt`):

- streamlit
- langchain
- langchain-core
- langchain-community
- langchain-google-genai
- langchain-groq
- faiss-cpu
- python-dotenv

---

## To Do / Future Enhancements

- Allow users to upload PDFs via the Streamlit interface
- Add chat history and memory
- Support multiple model options (OpenAI, Mistral, Claude, etc.)
- Save and load vector stores from disk
- Enhance UI with better design and usability
