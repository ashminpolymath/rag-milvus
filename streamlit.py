import streamlit as st
import requests
import json
import os

API_URL = "http://localhost:8000"  # FastAPI backend URL

# Page configuration
st.set_page_config(
    page_title="RAG Chat App",
    page_icon="🤖",
    layout="wide",
)


# Check backend status
with st.sidebar:
    st.header("System Status")

    # Check backend status only if explicitly requested
    if 'check_backend' not in st.session_state:
        st.session_state.check_backend = False

    check_status = st.button("Check Backend Status")

    if check_status or st.session_state.check_backend:
        st.session_state.check_backend = True
        try:
            # Only check a basic endpoint to see if server responds
            response = requests.get(f"{API_URL}/", timeout=2)
            st.success("✅ Backend is online and ready")
            backend_online = True
        except:
            st.error("❌ Cannot connect to backend")
            st.info("Make sure it's running with: `uvicorn main:app --reload`")
            backend_online = False

    # Session management
    st.header("Session Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

    # Show usage info
    st.header("Usage Info")
    st.info("""
    **How to use:**
    1. Upload documents in the "Upload Documents" tab
    2. Switch to "Chat" tab and ask questions
    3. The AI will answer based on your documents
    """)

    # Credits
    st.markdown("---")
    st.markdown("Built with Streamlit, FastAPI, and Milvus")

# Initialize session state variables
# Session state persists across Streamlit reruns
if 'messages' not in st.session_state:
    # Initialize with empty list for chat history
    st.session_state.messages = []

# Create tabs for upload and chat
tab1, tab2 = st.tabs(["Chat", "Upload Documents"])

# Chat interface tab
with tab1:
    st.header("Chat with RAG System")

    # Minimal CSS to keep chat input at the bottom
    st.markdown("""
    <style>
    /* Simple styles for the chat input container */
    .stChatFloatingInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        background-color: white !important;
        z-index: 999 !important;
        width: 100% !important;
        padding: 10px !important;
    }
    
    /* Add padding to ensure content isn't hidden */
    .main .block-container {
        padding-bottom: 80px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a container for the chat messages
    chat_container = st.container()

    # Display chat messages from history on app rerun
    # This ensures the chat history is preserved when the app reruns
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input widget - returns None until a user submits a message
    # This will be positioned at the bottom due to our CSS
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        # This shows the user's message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Display assistant response in chat message container
        with chat_container:
            with st.chat_message("assistant"):
                # Create a placeholder for streaming response text
                message_placeholder = st.empty()

                try:
                    # Prepare the payload according to the FastAPI backend's expected format
                    # We send the entire message history for context
                    payload = {
                        "messages": [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in st.session_state.messages
                        ]
                    }

                    # Show a spinner during API call
                    with st.spinner("Thinking..."):
                        # Make API call to FastAPI backend
                        response = requests.post(
                            f"{API_URL}/chat",
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=30  # Set a reasonable timeout for the chat response
                        )

                        # Process the response
                        if response.status_code == 200:
                            data = response.json()
                            full_response = data.get(
                                "data", "No response from the system.")
                        else:
                            full_response = f"Error: {response.text}"

                    # Display final response in the placeholder
                    message_placeholder.markdown(full_response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response})

                except requests.exceptions.ConnectionError:
                    # Handle connection errors specifically
                    error_msg = "Cannot connect to the backend server. Please make sure it's running."
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg})
                except Exception as e:
                    # Handle other exceptions
                    error_msg = f"An error occurred: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg})

# File upload tab
with tab2:
    st.header("Upload Documents to Knowledge Base")
    st.markdown(
        "Select one or more files to upload to the RAG system's knowledge base.")

    # File uploader widget
    # Accepts multiple files and restricts to common document formats
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "md"]  # Limit to text-based file types
    )

    # Process the upload when button is clicked and files are selected
    if uploaded_files and st.button("Upload to Knowledge Base"):
        with st.spinner("Uploading files..."):
            # Create a list to store uploaded files for the API request
            files_to_upload = []

            # Save uploaded files temporarily to disk
            # This is needed because we need to send the files as part of a multipart form
            for uploaded_file in uploaded_files:
                # Create temp directory if it doesn't exist
                os.makedirs("temp", exist_ok=True)

                # Save the file temporarily to disk
                temp_file_path = os.path.join("temp", uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Add to list of files to send in the multipart form
                # Format: (form field name, (filename, file object, content type))
                files_to_upload.append(
                    ("files", (uploaded_file.name, open(
                        temp_file_path, "rb"), "application/octet-stream"))
                )

            try:
                # Send files to backend using the /upload endpoint
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files_to_upload,
                    timeout=10  # Set a reasonable timeout
                )

                # Check if the upload was successful
                if response.status_code == 200:
                    st.success(
                        f"✅ {len(uploaded_files)} files uploaded successfully!")
                    # Display the response from the server
                    st.json(response.json())
                else:
                    st.error(f"❌ Error uploading files: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to the backend server. Please make sure it's running.")
                st.info(
                    "Run `uvicorn main:app --reload` in your terminal to start the backend.")
            except requests.exceptions.Timeout:
                st.error(
                    "❌ Request timed out. The server might be processing large files.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

            finally:
                # Resource cleanup - very important to avoid file handle leaks

                # Close all open file handles
                for file_tuple in files_to_upload:
                    file_tuple[1][1].close()

                # Clean up temporary files from disk
                for uploaded_file in uploaded_files:
                    temp_file_path = os.path.join("temp", uploaded_file.name)
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
