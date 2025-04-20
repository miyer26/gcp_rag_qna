import os
import logging
from google.cloud import aiplatform, storage
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextEmbeddingModel
from utils.chat_model import HuggingFaceQnA
import yaml

from google.cloud import aiplatform_v1
from google.api_core.client_options import ClientOptions

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VectorstoreQA:
    def __init__(self, gcp_details: dict, bucket_name: str, file_name: str):
        
        self.project = gcp_details['project_name']
        self.location = gcp_details['location']
        self.service_account_file = gcp_details['service_account_file']
        self.gcp_json_path = f"gs://{bucket_name}/embeddings/{file_name}"
        self.index_display_name = gcp_details.get("index_display_name", "japan_info_endpoint")
        self.deployed_index_id = gcp_details.get("deployed_index_id", "japan_info_endpoint_v1")


        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.service_account_file
        self.credentials = service_account.Credentials.from_service_account_file(self.service_account_file)

        aiplatform.init(project=self.project, location=self.location, credentials=self.credentials)
        vertexai.init(project=self.project, location=self.location)

        endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
        for ep in endpoints:
            if ep.display_name == self.index_display_name:
                endpoint_resource_name = ep.resource_name
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_resource_name)
        self.embedding_model = TextEmbeddingModel.from_pretrained(gcp_details['embedding_model'])
        self.llm = HuggingFaceQnA()
        logger.info(f"VectorstoreQA initialized with embedding model '{self.embedding_model}' and LLM '{self.llm}'")

    def get_embedding(self, text: str) -> list:
        """Convert input text to an embedding using Vertex AI."""
        logger.info("Generating embedding.")
        embedding_response = self.embedding_model.get_embeddings([text])
        return embedding_response[0].values

    def retrieve_similar_embeddings(self, query_embedding):
        """Retrieve top-K similar results from Matching Engine using low-level API."""
        logger.info("Retrieving similar embeddings using MatchServiceClient.")
 
        try:
            
            neighbors = self.endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=
                    [query_embedding]
                ,
                num_neighbors=20,
                return_full_datapoint=True
            )
            logger.info(f"Found {len(neighbors)} neighbors.")
            return neighbors[0].neighbors
        except Exception as e:
            logger.error(f"Error in find_neighbors request: {e}")
            return []

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using Hugging Face QnA model."""
        return self.llm.generate_answer(question, context)

    def answer_question(self, question: str) -> str:
        """Main QA pipeline: embed, retrieve, generate answer."""
        logger.info(f"Answering question: {question}")
        question_embedding = self.get_embedding(question)
        neighbors = self.retrieve_similar_embeddings(question_embedding)

        context_chunks = []
        for neighbor in neighbors:
            dp = neighbor.datapoint
            if dp and dp.datapoint_id:
                context_chunks.append(dp.datapoint_id)
        context = "\n".join(context_chunks)

        print(context)

        # Call the modified `generate_answer` method to use Hugging Face for the answer
        return self.generate_answer(question, context)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QnA using RAG")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    qna = VectorstoreQA(
            config['gcp_details'],
    config['bucket_name'],
    config['run_parameters']['output_file_name'],
    ) 
    print(qna.answer_question(config["question"]))

