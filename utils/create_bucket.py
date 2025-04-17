import os
import logging
from google.cloud import storage
from google.oauth2 import service_account
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BucketManager:
    def __init__(self, gcp_details, bucket_name):
        self.project = gcp_details['project_name']
        self.location = gcp_details['location']
        self.service_account_file = gcp_details['service_account_file']
        self.bucket_name = bucket_name

        # Authenticate using service account
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.service_account_file
        self.credentials = service_account.Credentials.from_service_account_file(self.service_account_file)

        self.client = storage.Client(project=self.project, credentials=self.credentials)

    def create_bucket(self):
        logger.info(f"Creating bucket: {self.bucket_name}")
        bucket = self.client.bucket(self.bucket_name)

        try:
            bucket.reload()
            logger.info(f"Bucket '{self.bucket_name}' already exists.")
        except Exception:
            new_bucket = self.client.create_bucket(bucket, location=self.location)
            logger.info(f"Bucket '{new_bucket.name}' created in {new_bucket.location}.")
            return new_bucket

        return bucket

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create GCP bucket")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    manager = BucketManager(config['gcp_details'], config['bucket_name'])
    manager.create_bucket()