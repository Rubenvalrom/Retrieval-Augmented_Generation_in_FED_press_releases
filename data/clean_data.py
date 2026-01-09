from langchain_community.document_loaders import PyPDFDirectoryLoader 
import os
import joblib
import spacy

INPUT_DIR = "raw"
OUTPUT_DIR = "clean"
METADATA_TO_KEEP = ['creationdate', 'total_pages', 'page']

# If the output directory doesn't exist, creates it
def setup_directory(output_dir=OUTPUT_DIR):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

# Download PDFs from the input directory
def download_pdfs(input_dir=INPUT_DIR):
    loader = PyPDFDirectoryLoader(input_dir) 
    raw_documents = loader.load()   
    return raw_documents


def remove_first_last_line(raw_document):
    """
    Remove the first and last lines from the page content of the document.
    """
    text = raw_document.page_content
    # Split into lines preserving order
    lines = text.splitlines()
    # If there are more than 2 lines, remove the first and last
    if len(lines) > 2:
        new_text = "\n".join(lines[1:-1]).strip()
    else:
        # If by some error there are 2 or fewer lines, leave empty
        new_text = ""
    raw_document.page_content = new_text
    return raw_document

def filter_lemmatize(raw_document, nlp): # Pass the whole object
    """
    Remove stop words and lemmatize, then update the document content.
    """
    # Process the text content
    spacy_doc = nlp(raw_document.page_content)

    # Filter and lemmatize  
    filtered_tokens = [token for token in spacy_doc if not token.is_stop]
    lemmatized_text = " ".join([token.lemma_.lower().strip() for token in filtered_tokens])
    
    # Update the object's content and return the whole object
    raw_document.page_content = lemmatized_text
    return raw_document


# Clean metadata to keep only relevant fields
def clean_metadata(raw_document, metadata_to_keep=METADATA_TO_KEEP):
    """
    Clean metadata to keep only relevant fields.
    """
    cleaned_metadata = {key: value for key, value in raw_document.metadata.items() if key in metadata_to_keep}
    raw_document.metadata = cleaned_metadata
    return raw_document

def clean_data(output_dir=OUTPUT_DIR, input_dir=INPUT_DIR):
    # Execute the data cleaning process
    setup_directory(output_dir)
    raw_documents = download_pdfs(input_dir)
    print(f"Total number of pages loaded: {len(raw_documents)}")

    # Remove first and last lines from each document
    cutted_documents = [remove_first_last_line(doc) for doc in raw_documents]
    print("First and last lines removed from each document.")

    # Load spaCy model (just the tokenizer and lemmatizer)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    print("spaCy model loaded.")

    # Keep negative words
    negative_words = ["no", "not", "nor", "n't", "never", "none", "nobody", "nothing", "neither", "without", "nevertheless"]

    for word in negative_words:
        nlp.vocab[word].is_stop = False

    # Remove stop words and lemmatize
    root_documents = [filter_lemmatize(doc, nlp) for doc in cutted_documents]
    print("Stop words removed and lemmatization applied.")

    # Clean metadata
    clean_documents = [clean_metadata(doc) for doc in root_documents]
    print("Metadata cleaned.")
    
    # Save cleaned documents in a pickle file
    joblib.dump(clean_documents, os.path.join(output_dir, 'clean_documents.pkl'))

    print(f"Cleaned documents saved to '{output_dir}/clean_documents.pkl'")

if __name__ == "__main__":
    clean_data()