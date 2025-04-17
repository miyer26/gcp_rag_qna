# file_uploader.py
import os
import logging
from google.cloud import storage
from google.oauth2 import service_account
import yaml
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GCSUploader:
    def __init__(self, gcp_details: str, bucket_name: str):
        self.project = gcp_details['project_name']
        self.location = gcp_details['location']
        self.service_account_file = gcp_details['service_account_file']
        self.bucket_name = bucket_name

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.service_account_file
        self.credentials = service_account.Credentials.from_service_account_file(self.service_account_file)

        self.client = storage.Client(project=self.project, credentials=self.credentials)
        self.bucket = self.client.bucket(self.bucket_name)

    def upload_file(self, local_path: str, destination_blob_name: str):
        logger.info(f"Uploading {local_path} to gs://{self.bucket_name}/{destination_blob_name}")
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        logger.info("Upload complete.")
        return f"gs://{self.bucket_name}/{destination_blob_name}"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Upload JSON to GCP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    uploader = GCSUploader(config['gcp_details'], config['bucket_name'])
    local_path = os.path.join(config['run_parameters']['output_path'], config['run_parameters']['output_file_name'])  # your local JSONL file path
    destination_name = f"embeddings/{config['run_parameters']['output_file_name']}"  # path within the bucket
    uploader.upload_file(local_path, destination_name)
