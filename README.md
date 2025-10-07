# RAG Application with Milvus Vector Database

This repository contains a Retrieval-Augmented Generation (RAG) application that uses Milvus as a vector database for document storage and retrieval. The application consists of a FastAPI backend for processing and storing documents, and a Streamlit frontend for user interaction.

## Architecture

The application has three main components:

1. **Docker Containers**: Milvus vector database and its dependencies
2. **FastAPI Backend**: Handles document processing, vector storage, and RAG-based chat responses
3. **Streamlit Frontend**: Provides a user interface for document upload and chat interaction

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- Python 3.8+ (Python 3.13 recommended)
- pip (Python package manager)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-milvus-backend
```

### 2. Start the Milvus Vector Database with Docker Compose

The application uses Docker Compose to set up Milvus and its required services (Etcd, MinIO).

```bash
docker-compose up -d
```

This will start the following services:
- Milvus: Vector database for storing document embeddings
- Etcd: Key-value store for Milvus metadata
- MinIO: Object storage for Milvus data

To verify that the services are running:

```bash
docker-compose ps
```

You should see all services in the "Up" state.

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a .env File

Create a `.env` file in the root directory of the project by copying the example file:

```bash
# On Windows
copy .env.example .env

# On Unix/Linux/Mac
cp .env.example .env
```

Then edit the `.env` file and update the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_STORE_URI=http://localhost:19530
VECTOR_STORE_COLLECTION_NAME=documents
```

Replace `your_openai_api_key_here` with your actual OpenAI API key. This key is required for the AI chat functionality.

The `.env` file is used to store sensitive configuration data and environment-specific settings. The application uses the python-dotenv package to load these variables from this file.

### 5. Start the FastAPI Backend

The backend handles document processing, vector storage, and RAG-based chat responses.

```bash
uvicorn main:app --reload
```

The API will be available at: http://localhost:8000

You can access the API documentation at: http://localhost:8000/docs

### 6. Start the Streamlit Frontend

The frontend provides a user interface for document upload and chat interaction.

```bash
streamlit run streamlit.py
```

The Streamlit app will be available at: http://localhost:8501

## Using the Application

### Uploading Documents

1. Navigate to the "Upload Documents" tab
2. Select one or more files (supported formats: TXT, PDF, DOCX, MD)
3. Click the "Upload to Knowledge Base" button
4. Wait for the confirmation message

### Chatting with the RAG System

1. Navigate to the "Chat" tab
2. Type your question in the input field at the bottom
3. The system will retrieve relevant information from your documents and provide a response

## Stopping the Application

1. Stop the Streamlit app by pressing Ctrl+C in its terminal
2. Stop the FastAPI server by pressing Ctrl+C in its terminal
3. Stop the Docker containers:

```bash
docker-compose down
```

To completely remove all data and start fresh:

```bash
docker-compose down -v
```

## Troubleshooting

### Backend Connection Issues

If the Streamlit app cannot connect to the backend:
1. Make sure the FastAPI backend is running
2. Check that it's running on port 8000 (default)
3. Use the "Check Backend Status" button in the sidebar to verify the connection

### Milvus Connection Issues

If the backend cannot connect to Milvus:
1. Check if all Docker containers are running: `docker-compose ps`
2. Restart the containers if needed: `docker-compose restart`
3. Check the logs for any errors: `docker-compose logs milvus`
4. Verify the `VECTOR_STORE_URI` in your `.env` file matches the Milvus endpoint (default: `http://localhost:19530`)

### Environment Variable Issues

If you encounter errors related to missing environment variables:
1. Make sure you've created the `.env` file as described in step 4
2. Check if the `.env` file is in the root directory of the project
3. Verify that all required variables are set in the `.env` file
4. Restart the backend server after making changes to the `.env` file

## File Structure

- `main.py`: FastAPI backend implementation
- `store.py`: Milvus vector store interface
- `ai.py`: AI client for RAG-based responses
- `models.py`: Data models for the application
- `streamlit.py`: Streamlit frontend
- `config.py`: Configuration settings
- `docker-compose.yml`: Docker Compose configuration for Milvus
- `.env`: Environment variables (must be created based on .env.example)
- `.env.example`: Example environment variables file

## Contributing

Please read the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.