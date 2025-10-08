from pymilvus import MilvusClient
from ai import ai_client
from config import settings
import PyPDF2


# Milvus Vector Format
# { id: int, vector: List[float], "text": str }

class MilvusStore:
    def __init__(self, uri: str, collection_name: str):
        self.client = MilvusClient(uri)
        self.collection_name = collection_name

        # Ensure the collection exists
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                self.collection_name, dimension=1536, metric_type="COSINE")

    def upsert_file(self, file_name: str, file_path: str):
        # Handle different file types
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for i, page in enumerate(pdf_reader.pages):
                    content = page.extract_text()
                    embedding = ai_client.embed_text(content)
                    data = [{
                        "id": hash(file_name + str(i)),
                        "vector": embedding,
                        "text": content,
                        "page_number": i + 1,
                        "file_name": file_name
                    }]
                    self.client.insert(
                        collection_name=self.collection_name, data=data)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                embedding = ai_client.embed_text(content)
                data = [{
                    "id": hash(file_name),
                    "vector": embedding,
                    "text": content,
                    "file_name": file_name
                }]
                self.client.insert(
                    collection_name=self.collection_name, data=data)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        self.client.flush(self.collection_name)

    def search_query(self, question: str, top_k: int):
        retrieved_data = self.client.search(collection_name=self.collection_name, data=[ai_client.embed_text(
            question)], limit=top_k, search_params={"metric_type": "COSINE", "params": {}}, output_fields=["text", "page_number", "file_name"])
        return retrieved_data


milvus_store = MilvusStore(
    uri=settings.VECTOR_STORE_URI,
    collection_name=settings.VECTOR_STORE_COLLECTION_NAME
)
