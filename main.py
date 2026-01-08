from src.rag import rag
import gradio as gr

def generate_response(query):
    """
    Invokes the RAG pipeline to get a structured response.
    """
    generated_answer = rag(query)
    return generated_answer

def get_field(generated_answer, field_name: str) -> str:
    """
    Extracts specific fields (Answer, Sentiment, Evidence) from the 
    JSON returned by the LLM.
    """
    keys = generated_answer.keys()
    # Search for keys that contain the field name (case-insensitive)
    answer_keys = [k for k in keys if field_name.lower() in k.lower()]

    if not answer_keys:
        return "Information not found in the context."
    elif len(answer_keys) == 1:
        return str(generated_answer[answer_keys[0]])
    else:
        # Concatenate multiple matches if they exist
        return "\n".join([str(generated_answer[k]) for k in answer_keys])

def pipeline(query):
    """
    Orchestrates the retrieval and formatting for the Gradio UI.
    Returns: (Sentiment, Answer)
    """
    try:
        # Generate the raw JSON response from the RAG chain
        raw_output = generate_response(query)
        
        # Extract Sentiment and Answer fields
        sentiment = get_field(raw_output, "Sentiment")
        answer = get_field(raw_output, "Answer")
        
        return sentiment, answer
    except Exception as e:
        return f"Error processing the query: {e}"

def launch_interface():
    """
    Launches the Gradio interface.
    - One large input text field for the question.
    - Two output fields:
        - Sentiment (short)
        - Answer (large)
    """
    with gr.Blocks(title="FED sentiment analysis with RAG") as demo:

        gr.Markdown("## FED sentiment analysis with RAG")
        gr.Markdown("Data goes from 2011 to 2025 Q4.")

        # Input
        query_input = gr.Textbox(
            label="Question",
            placeholder="Enter your question here...",
            lines=3
        )

        # Outputs
        sentiment_output = gr.Textbox(
            label="Sentiment",
            lines=1,
            placeholder="Dovish / Neutral / Hawkish",
            interactive=False
        )

        answer_output = gr.Textbox(
            label="Answer",
            lines=12,
            interactive=False
        )

        submit_btn = gr.Button("Submit")

        submit_btn.click(
            fn=pipeline,
            inputs=query_input,
            outputs=[sentiment_output, answer_output]
        )

    demo.launch()


if __name__ == "__main__":
    launch_interface()