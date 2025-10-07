from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, HTTPException
import os
from models import MessagesPayload
from store import milvus_store
from ai import ai_client

app = FastAPI(
    title="Milvus Vector Store API",
    description="API for interacting with the Milvus vector store",
    version="1.0.0",
    root_path="/milvus_vector_store",
)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"status": "ok", "message": "Milvus Vector Store API is running"}


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload multiple files to the server"""
    uploaded_files = []

    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files selected")

        # Create uploads directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        for file in files:
            if not file.filename:
                continue

            # Save each file
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            uploaded_files.append({
                "filename": file.filename,
                "file_path": file_path,
            })

            # Process and store the file in Milvus
            milvus_store.upsert_file(file.filename, file_path)

        return {
            "message": f"{len(uploaded_files)} files uploaded and processed successfully",
            "files": uploaded_files
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error uploading files: {str(e)}")

    finally:
        for file_info in uploaded_files:
            try:
                if os.path.exists(file_info["file_path"]):
                    os.remove(file_info["file_path"])
            except Exception as cleanup_error:
                print(
                    f"Warning: Could not delete file {file_info['file_path']}: {cleanup_error}")


@app.post("/chat")
async def chat(payload: MessagesPayload):
    """Handle chat messages and return a response"""
    try:
        if not payload.messages or len(payload.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Extract the latest user message
        user_message = next(
            (msg for msg in reversed(payload.messages) if msg.role == "user"), None)

        if not user_message:
            raise HTTPException(
                status_code=400, detail="No user message found in the payload")

        # Search for relevant documents in Milvus
        search_results = milvus_store.search_query(
            question=user_message.content, top_k=3)
        print("Search Results:", search_results)

        # Compile the response with retrieved documents
        retrieved_lines_with_distances = [
            (res["entity"]["text"], res["distance"]) for res in search_results[0]
        ]

        context = "\n".join(
            [line_with_distance[0]
                for line_with_distance in retrieved_lines_with_distances]
        )

        response = ai_client.chat_with_rag(
            context=context,
            question=user_message.content,
            history=[msg.content for msg in payload.messages[:-1]]
        )

        return {"data": response}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat: {str(e)}")
