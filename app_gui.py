"""
Gradio GUI for Legal Policy Explainer.
wraps the local LLM agent in a web interface.
"""

import gradio as gr
import logging
import argparse
from pathlib import Path
from app_local import load_config, setup_logging, initialize_system

# Global agent variable
agent = None


def chat_response(message, history):
    """
    Wrapper for agent query.
    """
    global agent
    if not agent:
        return "⚠️ System not initialized. Please check the server logs."

    try:
        # The agent is stateless, so we simply pass the new message.
        # History is handled by the ChatInterface UI, but the agent
        # treats each query independently (as per current implementation).
        response = agent.query(message)
        return response
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return f"❌ Error: {str(e)}"


def format_context(history):
    """
    Optional: Format history for the agent if we wanted to support multi-turn.
    Currently unused as UnifiedLegalAgent is stateless.
    """
    return history


def create_ui():
    """Create the Gradio interface with custom styling."""

    # Custom CSS for a more premium, dark-themed look
    custom_css = """
    .gradio-container {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    h1 {
        color: #38bdf8; /* Sky 400 */
        font-weight: 800;
    }
    .chat-message {
        border-radius: 12px;
    }
    """

    # Define a clean, professional theme
    theme = gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ).set(
        body_background_fill="slate-950",
        block_background_fill="slate-900",
        block_border_width="1px",
        block_border_color="slate-800",
        button_primary_background_fill="cyan-600",
        button_primary_background_fill_hover="cyan-500",
        button_primary_text_color="white",
    )

    # Create Blocks with theme
    with gr.Blocks(title="⚖️ Legal Policy Explainer") as demo:
        gr.ChatInterface(
            fn=chat_response,
            title="⚖️ Legal Policy Explainer",
            description="""
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
                <p style="font-size: 1.1em; color: #94a3b8;">
                    AI-powered legal assistant running 100% locally. 
                    Ask questions about regulations, contracts, and policies.
                </p>
            </div>
            """,
            examples=[
                "What is a non-disclosure agreement?",
                "Explain the key elements of a valid contract?",
                "What are the implications of copyright infringement?",
                "Define 'force majeure' in the context of business contracts.",
            ],
        )

    return demo, theme, custom_css


def main():
    global agent

    parser = argparse.ArgumentParser(description="Legal Policy Explainer GUI")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_local.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--no-rag", action="store_true", help="Disable RAG functionality"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public share link"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the server on"
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config_path = args.config
        if not Path(config_path).exists():
            print(f"Config file not found: {config_path}")
            # Fallback to default if custom path fails
            config_path = "config/config_local.yaml"

        config = load_config(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Setup logging
    setup_logging(config)

    # Initialize system
    try:
        print("Starting Legal Policy Explainer GUI...")
        agent = initialize_system(config, enable_rag=not args.no_rag)
    except Exception as e:
        print(f"Failed to initialize system: {e}")
        return

    # Create and launch UI
    ui, theme, css = create_ui()

    print(f"Launching Gradio interface on port {args.port}...")
    ui.queue().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        favicon_path=None,
        theme=theme,
        css=css,
    )


if __name__ == "__main__":
    main()
