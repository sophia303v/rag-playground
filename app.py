"""Gradio UI for RAG Playground."""
import sys
sys.path.insert(0, ".")

import gradio as gr
from PIL import Image
from src.rag_pipeline import RAGPipeline
from src.causal_chain import (
    extract_causal_chains,
    get_default_system_prompt,
    get_default_user_prompt,
)

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


def analyze_causal_chains(
    article: str,
    system_prompt: str,
    user_prompt_template: str,
) -> tuple[str, str]:
    """
    Extract causal chains from an article using user-provided prompts.

    Returns:
        tuple of (formatted_output, raw_llm_response)
    """
    if not article.strip():
        return "Please paste an article or text.", ""

    try:
        result, raw_response = extract_causal_chains(
            article,
            system_prompt=system_prompt.strip() or None,
            user_prompt_template=user_prompt_template.strip() or None,
        )

        # If we got structured JSON, render it nicely
        if result and isinstance(result, dict):
            output = _format_structured(result)
        else:
            # Otherwise just show the raw LLM response as-is
            output = raw_response

        return output, raw_response

    except Exception as e:
        return f"Error: {str(e)}", ""


def _format_structured(result: dict) -> str:
    """Best-effort formatting of a JSON result into Markdown."""
    parts = []

    # Summary
    summary = result.get("summary", "")
    if summary:
        parts.append(f"### Summary\n{summary}")

    # Causal pairs table
    pairs = result.get("causal_pairs", [])
    if pairs:
        rows = [
            "### Causal Pairs",
            "| # | Cause | Effect | Confidence | Evidence |",
            "|---|-------|--------|------------|----------|",
        ]
        for p in pairs:
            pid = p.get("id", "")
            cause = str(p.get("cause", "")).replace("|", "\\|")
            effect = str(p.get("effect", "")).replace("|", "\\|")
            conf = p.get("confidence", "")
            evidence = str(p.get("evidence", "")).replace("|", "\\|")
            if len(evidence) > 80:
                evidence = evidence[:77] + "..."
            rows.append(f"| {pid} | {cause} | {effect} | {conf} | {evidence} |")
        parts.append("\n".join(rows))

    # Causal chains
    chains = result.get("causal_chains", [])
    if chains:
        chain_lines = ["### Causal Chains"]
        for i, c in enumerate(chains, 1):
            nodes = c.get("chain", [])
            desc = c.get("description", "")
            chain_lines.append(f"**Chain {i}:** {' → '.join(nodes)}")
            if desc:
                chain_lines.append(f"  _{desc}_")
            chain_lines.append("")
        parts.append("\n".join(chain_lines))

    return "\n\n".join(parts) if parts else "(No structured output found)"


# --- Example articles ---
CAUSAL_CHAIN_EXAMPLES = [
    [
        "「如果不跟我親密接觸，我會想強暴你」\n"
        "一句話就可以看出他是什麼樣的人\n"
        "一是社交障礙 因為一般不會直接說出來\n"
        "社交障礙是 不知道自己這樣做了 對方會有什麼反應\n"
        "二是 缺乏良心\n"
        "「會想強暴」不是有良心 同理心 有克制能力 有內疚回饋的人會說的話"
    ],
    [
        "Climate change is causing glaciers to melt at an unprecedented rate. "
        "As glaciers melt, sea levels rise, threatening coastal communities. "
        "Rising sea levels lead to increased flooding in low-lying areas, "
        "which displaces populations and strains urban infrastructure."
    ],
    [
        "Overuse of antibiotics in livestock farming has accelerated the evolution of "
        "antibiotic-resistant bacteria. These resistant strains can transfer to humans "
        "through the food supply chain. As common antibiotics become ineffective, "
        "treating routine infections becomes more difficult and expensive."
    ],
]


# Build Gradio interface
with gr.Blocks(
    title="RAG Playground",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # RAG Playground
        """
    )

    with gr.Tabs():
        # --- Tab 1: RAG Query ---
        with gr.TabItem("RAG Query"):
            gr.Markdown(
                """
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

        # --- Tab 2: Causal Chain Extraction ---
        with gr.TabItem("Causal Chain Extraction"):
            gr.Markdown(
                """
                Paste text below and let LLM extract causal relationships.
                You can **edit the System Prompt and User Prompt** to control
                how the analysis is done — different prompts produce completely
                different results.
                """
            )

            # --- Prompt editing area (collapsible) ---
            with gr.Accordion("Prompt Design", open=False):
                causal_system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=get_default_system_prompt,
                    lines=4,
                )
                causal_user_prompt = gr.Textbox(
                    label="User Prompt Template (use {article} as placeholder for input text)",
                    value=get_default_user_prompt,
                    lines=18,
                )

            with gr.Row():
                with gr.Column(scale=1):
                    article_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Paste your article, conversation, or any text here...",
                        lines=12,
                    )
                    extract_btn = gr.Button(
                        "Extract Causal Chains",
                        variant="primary",
                        size="lg",
                    )

                    gr.Examples(
                        examples=CAUSAL_CHAIN_EXAMPLES,
                        inputs=[article_input],
                        label="Examples",
                    )

                with gr.Column(scale=1):
                    causal_output = gr.Markdown(label="Analysis Result")
                    with gr.Accordion("Raw LLM Response", open=False):
                        raw_output = gr.Code(
                            label="Raw Response",
                            language=None,
                        )

            extract_btn.click(
                fn=analyze_causal_chains,
                inputs=[article_input, causal_system_prompt, causal_user_prompt],
                outputs=[causal_output, raw_output],
            )
            article_input.submit(
                fn=analyze_causal_chains,
                inputs=[article_input, causal_system_prompt, causal_user_prompt],
                outputs=[causal_output, raw_output],
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
