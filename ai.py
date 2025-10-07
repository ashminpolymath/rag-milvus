from openai import OpenAI
from config import settings


class AIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> list:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def chat_with_rag(self, context: str, question: str, history: list[str]):
        SYSTEM_PROMPT = """
        Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
        """

        USER_PROMPT = f"""
        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
        <history>
        {history}
        </history>
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content


ai_client = AIClient(api_key=settings.OPENAI_API_KEY)
