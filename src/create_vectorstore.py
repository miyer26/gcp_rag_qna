#from vertexai.preview.vectorstores.matching_engine import MatchingEngineIndex
from google.cloud import aiplatform
from google.cloud import storage
from google.oauth2 import service_account
import os

import yaml
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Vectorstore:
    def __init__(self, gcp_details: str, bucket_name: str, file_name: str):
        self.project = gcp_details['project_name']
        self.location = gcp_details['location']
        self.service_account_file = gcp_details['service_account_file']
        self.gcp_json_path = f"gs://{bucket_name}/embeddings/{file_name}"
        self.index_display_name = ""
        self.deployed_index_id = ""

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.service_account_file
        logger.info(f"Initializing Vertex AI for project={self.project}, {self.location}")
        self.credentials = service_account.Credentials.from_service_account_file(self.service_account_file)

        self.client = storage.Client(project=self.project, credentials=self.credentials)
        aiplatform.init(project=self.project, location=self.location)

    def index_exists(self, display_name: str) -> bool:
        # Check if the index already exists
        try:
            indexes = aiplatform.MatchingEngineIndex.list(filter=f"display_name={display_name}")
            if indexes:
                logger.info(f"Index with display_name '{display_name}' already exists.")
                return True
            else:
                logger.info(f"No existing index found with display_name '{display_name}'.")
                return False
        except Exception as e:
            logger.error(f"Error while checking for existing index: {e}")
            return False
    
    def deployment_exists(self, endpoint: aiplatform.MatchingEngineIndexEndpoint, deployed_index_id: str) -> bool:
        # Check if the index is already deployed to the endpoint
        try:
            deployed_indexes = endpoint.list_deployed_indexes()
            for deployed_index in deployed_indexes:
                if deployed_index.id == deployed_index_id:
                    logger.info(f"Index with id '{deployed_index_id}' is already deployed.")
                    return True
            logger.info(f"No deployment found with id '{deployed_index_id}'.")
            return False
        except Exception as e:
            logger.error(f"Error while checking for existing deployment: {e}")
            return False
    
    def create_index(self):
        index_display_name = "japan_info_v1"

        if self.index_exists(index_display_name):
            logger.info(f"Skipping index creation since an index with the name '{index_display_name}' already exists.")
            return None
        # create index
        my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                        display_name=index_display_name,
                        contents_delta_uri=self.gcp_json_path,
                        dimensions=768,
                        approximate_neighbors_count=10,
                        distance_measure_type="DOT_PRODUCT_DISTANCE"
                    )
        logger.info("Index created successfully.")
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name="japan_info_endpoint",
            public_endpoint_enabled=True)
        
        deployed_index_id = "japan_info_endpoint_v1"

        if self.deployment_exists(endpoint, deployed_index_id):
            logger.info(f"Skipping deployment since the index with id '{deployed_index_id}' is already deployed.")
        else:
            # Deploy the index if not already deployed
            try:
                endpoint.deploy_index(index=my_index, deployed_index_id=deployed_index_id)
                logger.info("Index deployed successfully.")
            except Exception as e:
                logger.error(f"Error deploying index: {e}")
                return None

        return endpoint

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create Vectorstore on GCP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    vectorstore = Vectorstore(config["gcp_details"], config["bucket_name"], config["run_parameters"]["output_file_name"])
    vectorstore.create_index()
