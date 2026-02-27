"""Gradio UI for RAG Playground."""
import sys
sys.path.insert(0, ".")

import gradio as gr
from PIL import Image
from src.rag_pipeline import RAGPipeline

# Initialize RAG pipeline
print("Initializing RAG Playground...")
rag = RAGPipeline()

if not rag._is_ingested:
    print("No index found. Running ingestion...")
    rag.ingest(max_samples=300)


def query_rag(question: str, image: Image.Image | None = None) -> tuple[str, str]:
    """
    Process a query through the RAG pipeline.

    Returns:
        tuple of (answer, sources)
    """
    if not question.strip():
        return "Please enter a question.", ""

    try:
        result = rag.query(question, image=image)

        # Format sources
        sources_text = "### Retrieved Sources\n\n"
        for i, (doc, meta, dist) in enumerate(zip(
            result.retrieval.documents,
            result.retrieval.metadatas,
            result.retrieval.distances,
        )):
            relevance = f"{(1 - dist) * 100:.1f}%"
            uid = meta.get("uid", "unknown")
            section = meta.get("section", "unknown")
            sources_text += f"**[Source {i+1}]** Report {uid} ({section}) - Relevance: {relevance}\n"
            sources_text += f"> {doc[:200]}...\n\n"

        if result.retrieval.image_description:
            sources_text += f"\n### Image Analysis\n{result.retrieval.image_description}"

        return result.answer, sources_text

    except Exception as e:
        return f"Error: {str(e)}", ""


# Build Gradio interface
with gr.Blocks(
    title="RAG Playground",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # RAG Playground

        Ask questions over your document corpus. Optionally upload an image
        for multimodal analysis.

        *This system retrieves relevant documents and uses AI to generate
        evidence-based answers with source citations.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload Medical Image (optional)",
                type="pil",
                height=300,
            )
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What does cardiomegaly look like on a chest X-ray?",
                lines=3,
            )
            submit_btn = gr.Button("Ask", variant="primary", size="lg")

            gr.Examples(
                examples=[
                    ["What are the most common findings in chest X-rays?"],
                    ["Describe the typical appearance of pneumonia on chest radiograph."],
                    ["What does cardiomegaly indicate and how is it identified?"],
                    ["What are the signs of pleural effusion on X-ray?"],
                    ["How can you differentiate between consolidation and atelectasis?"],
                ],
                inputs=[question_input],
                label="Example Questions",
            )

        with gr.Column(scale=1):
            answer_output = gr.Markdown(label="AI Answer")
            sources_output = gr.Markdown(label="Sources & Evidence")

    # Connect the interface
    submit_btn.click(
        fn=query_rag,
        inputs=[question_input, image_input],
        outputs=[answer_output, sources_output],
    )
    question_input.submit(
        fn=query_rag,
        inputs=[question_input, image_input],
        outputs=[answer_output, sources_output],
    )

    gr.Markdown(
        """
        ---
        *Disclaimer: This tool is for educational and research purposes only.
        It should not be used for clinical diagnosis or treatment decisions.
        Always consult qualified medical professionals.*
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
