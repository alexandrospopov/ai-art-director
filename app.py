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

    # Find all images in temp_dir (sorted by name)
    image_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith((".jpg", ".png"))])
    images = [Image.open(p) for p in image_files]

    yield images, "Enhancement finished."


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
            gallery = gr.Gallery(label="Image Progress", show_label=True)
            agent_logs = gr.Textbox(label="Agent Logs", lines=18, interactive=False)

    submit_btn.click(
        process_image_with_agents,
        inputs=[image_input, prompt_input],
        outputs=[gallery, agent_logs],
    )

    demo.queue()

if __name__ == "__main__":
    demo.launch()
