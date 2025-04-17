import os
import logging
import argparse
from typing import List, Dict

from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.auth
from google.auth.transport.requests import Request

import vertexai
from vertexai.language_models import TextEmbeddingModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Embedder():
    def __init__(self, gcp_details: Dict, chunking_params: Dict):
        project = gcp_details['project_name']
        location = gcp_details['location']
        model_name = gcp_details['embedding_model']
        service_account_file = gcp_details['service_account_file']
        logger.info(f"Initializing Vertex AI for project={project}, location={location}")
        
        # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the service account key
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
        
        # Authenticate with the service account
        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())


        vertexai.init(project=project, location=location)
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.chunk_size = chunking_params['chunk_size']
        self.chunk_overlap = chunking_params['chunk_overlap']
    
    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        logger.info(f"Extracting text from: {pdf_path}")
        reader = PdfReader(pdf_path)
        return [page.extract_text() or "" for page in reader.pages]

    def chunk_text(self, texts_by_page: List[str]) -> List[Dict]:
        logger.info("Chunking text with metadata...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        all_chunks = []
        for page_num, text in enumerate(texts_by_page):
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"chunk_{page_num}_{i}",
                    "text": chunk,
                    "source": [],
                    "page": page_num + 1
                })
        return all_chunks

    def embed_chunks(self, chunk_data: List[Dict]) -> List[Dict]:
        logger.info(f"Generating embeddings for {len(chunk_data)} chunks...")
        result = []
        for data in tqdm(chunk_data, desc="Embedding"):
            embedding = self.model.get_embeddings([data["text"]])[0].values
            record = {
                "id": data["id"],
                "embedding": embedding,
                "restricts": [],
                "crowding_tag": "",
                "metadata": {
                    "text": data["text"],
                    "source": data["source"],
                    "page": data["page"]
                }
            }
            result.append(record)
        return result


    def save_embeddings_jsonl(self, chunks: List[Dict], embeddings: List[List[float]], output_path: str, output_file_name: str):
        logger.info(f"Saving embeddings to {output_path}")

        if not os.path.exists(output_path):
            logger.info(f"Creating {output_path}")
            os.makedirs(output_path)

        output_file = os.path.join(output_path, output_file_name)
        with open(output_file, "w") as f:
            for chunk, emb in zip(chunks, embeddings):
                record = {
                    "id": chunk.get("id"),
                    "embedding": emb,
                    "metadata": {
                        "text": chunk["text"],
                        "source": chunk.get("source"),
                        "page": chunk.get("page")
                    }
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Saved {len(chunks)} embeddings to {output_file_name} in JSONL format.")
    
    def run(self, pdf_path: str, output_path: str, output_file_name: str):

        text = self.extract_text_by_page(pdf_path) 
        chunks = self.chunk_text(text)
        embeddings = self.embed_chunks(chunks)
        self.save_embeddings_jsonl(chunks, embeddings, output_path, output_file_name) 
        logger.info("✅ Done embedding PDF!")
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDF embeddings with Vertex AI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    embedder = Embedder(config["gcp_details"], config["chunking_parameters"])
    embedder.run(config["run_parameters"]["input_file"], config["run_parameters"]["output_path"], config["run_parameters"]["output_file_name"])