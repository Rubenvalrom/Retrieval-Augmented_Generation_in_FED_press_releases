from langchain_community.document_loaders import PyPDFDirectoryLoader 
import os
from urllib.parse import urljoin
import joblib

INPUT_DIR = "raw"
OUTPUT_DIR = "clean"
METADATA_TO_KEEP = ['creationdate', 'total_pages', 'page']

# If the output directory doesn't exist, creates it
def setup_directory():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directory '{OUTPUT_DIR}' created.")

# Download PDFs from the input directory
def download_pdfs():
    loader = PyPDFDirectoryLoader(INPUT_DIR) 
    raw_documents = loader.load()   
    return raw_documents

# Clean metadata to keep only relevant fields
def clean_metadata(raw_document):
    cleaned_metadata = {key: value for key, value in raw_document.metadata.items() if key in METADATA_TO_KEEP}
    raw_document.metadata = cleaned_metadata
    return raw_document

# Execute the data cleaning process
def main():
    setup_directory()
    raw_documents = download_pdfs()
    print(f"Total number of pages loaded: {len(raw_documents)} páginas de PDF en total.")

    clean_documents = [clean_metadata(doc) for doc in raw_documents]

    # Save cleaned documents in a pickle file
    joblib.dump(clean_documents, os.path.join(OUTPUT_DIR, 'clean_documents.pkl'))

    print(f"Cleaned documents saved to '{OUTPUT_DIR}/clean_documents.pkl'")

if __name__ == "__main__":
    main()