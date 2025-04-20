import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class HuggingFaceQnA:

    def __init__(self, model_name: str = "google/flan-t5-base"):

        load_dotenv()
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face token not found in .env file.")
        self.client = InferenceClient(model=model_name, token=hf_token)

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Hugging Face's hosted model."""
        logger.info("Calling Hugging Face LLM for answer.")

        prompt = f"""You are a helpful assistant.
        Given the context below, answer the following question.

        Context:
        {context}

        Question:
        {question}

        Provide a concise, clear, and direct answer to the question based on the context.
        Answer:"""

        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=250,
                temperature=0.3,
                top_p=0.8,
                do_sample=True,
                stop=["###"]
            )
            answer = response.strip()
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return ""


if __name__ == "__main__":
    hf = HuggingFaceQnA()
    print(hf.generate_answer("How many cars are red?", "Three of the cars are red and 2 are blue"))