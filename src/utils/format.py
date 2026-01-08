import json_repair
from langchain_core.output_parsers import JsonOutputParser

def parse_with_fixer(text):
    parser = JsonOutputParser()
    
    try:
        # Normal parsing attempt
        json_output = parser.parse(text)

        return json_output
    
    except Exception as e:
        # Fallback using json_repair
        json_fixed = json_repair.loads(text)
        
        return json_fixed
    

def format_docs(docs):
    formatted = []
    # Iterate through documents, extract metadata and content
    for doc in docs:
        # Extract metadata
        meta = doc.metadata
        date = meta.get('creationdate', 'Unknown Date')
        page = meta.get('page', '?')
        total_pages = meta.get('total_pages', '?')

        # Clean content by replacing newlines with spaces
        content = doc.page_content.replace("\n", " ")

        formatted.append(f"FRAGMENT [Date: {date} | Page: {page} of {total_pages}] \n{content}")

    # Combine all formatted documents into a single context string separated by double newlines
    context = "\n\n".join(formatted)

    return context