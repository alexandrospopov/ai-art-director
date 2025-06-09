import os
import tempfile

import gradio as gr
from PIL import Image

from agents import run_photo_enchancement_agent


def process_image_with_agents(image: Image.Image, prompt: str):
    temp_dir = tempfile.mkdtemp(prefix="gradio_aiart_")
    input_path = os.path.join(temp_dir, "input.jpg")
    output_directory = temp_dir
    image.save(input_path)

    yield [image], "Original image uploaded. Starting enhancement…"

    run_photo_enchancement_agent(
        prompt,
        image_path=input_path,
        output_directory=output_directory,
    )

    image_files = sorted(
        [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".jpeg")], key=os.path.getmtime
    )
    images = [(Image.open(p), os.path.basename(p)) for p in image_files]

    yield images, "✅ Enhancement finished."


with gr.Blocks(title="AI Art Director • Agent Workflow") as demo:
    gr.Markdown(
        "# AI Art Director\n"
        "Upload an image and describe the vibe you want.\n"
        "The agent will propose, apply, and critique edits to match your vision.\n"
        "🕒 Disclaimer: The agent takes a LONG time (around 10 minutes).\n"
        "📜 You can follow the activity through logs."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(label="Describe the vibe you want", placeholder="e.g. dreamy, vintage, vibrant…")
            submit_btn = gr.Button("Go!")
        with gr.Column():
            gallery = gr.Gallery(label="Image Progress", show_label=True)
            status_box = gr.Textbox(label="Status", lines=2, interactive=False)

    submit_btn.click(
        process_image_with_agents,
        inputs=[image_input, prompt_input],
        outputs=[gallery, status_box],  # ✅ Fixed: include both outputs
    )

    demo.queue()

if __name__ == "__main__":
    demo.launch()
