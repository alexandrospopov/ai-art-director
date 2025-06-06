import contextlib
import io
import os
import tempfile

import gradio as gr
from PIL import Image

from agents import run_photo_enchancement_agent


def process_image_with_agents(image: Image.Image, prompt: str):
    """Stream intermediate steps **and** the agent's stdout / stderr logs."""
    # 🔧 1. Create temp dir & paths
    temp_dir = tempfile.mkdtemp(prefix="gradio_aiart_")
    input_path = os.path.join(temp_dir, "input.jpg")
    output_path = os.path.join(temp_dir, "output.jpg")

    # 💾 2. Persist original upload
    image.save(input_path)

    # 🖼️ 3. Yield the original image immediately
    yield image, "Original image uploaded. Starting enhancement…"

    # 📝 4. Capture logs while the agent runs
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
        _ = run_photo_enchancement_agent(
            prompt,
            image_path=input_path,
            output_path=output_path,
        )

    # 🧾 All logs produced by the agent
    logs = log_buffer.getvalue()

    # 🖼️ 5. Yield the final image plus the complete logs
    final_image = Image.open(output_path)
    yield final_image, f"✅ Enhancement finished.\n\n--- Agent Logs ---\n{logs}"


with gr.Blocks(title="AI Art Director • Agent Workflow") as demo:
    gr.Markdown(
        "# AI Art Director\n"
        "Upload an image and describe the vibe you want.\n"
        "The agent will propose, apply, and critique edits to match your vision "
        "– and you'll see progress **and logs** live!"
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(label="Describe the vibe you want", placeholder="e.g. dreamy, vintage, vibrant…")
            submit_btn = gr.Button("Go!")
        with gr.Column():
            streamed_image = gr.Image(label="Image Progress")
            agent_logs = gr.Textbox(label="Agent Logs", lines=18, interactive=False)

    submit_btn.click(
        process_image_with_agents,
        inputs=[image_input, prompt_input],
        outputs=[streamed_image, agent_logs],
    )

    # Allow multiple users to queue without blocking streaming
    demo.queue()

if __name__ == "__main__":
    demo.launch()
