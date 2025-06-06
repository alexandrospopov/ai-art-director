import os
import tempfile

import gradio as gr
from PIL import Image

from agents import run_photo_enchancement_agent


def process_image_with_agents(image, prompt):
    # Save uploaded image to a temp file
    temp_dir = tempfile.mkdtemp(prefix="gradio_aiart_")
    input_path = os.path.join(temp_dir, "input.jpg")
    image.save(input_path)

    output_path = os.path.join(temp_dir, "output.jpg")

    # Simulate streaming: first yield original image
    yield [[input_path], ["Original"], image]

    # Run the enhancement agent
    _ = run_photo_enchancement_agent(prompt, image_path=input_path, output_path=output_path)

    # Load the final image
    final_image = Image.open(output_path)

    # Stream final result
    yield [[input_path, output_path], ["Original", "Final (after agent workflow)"], final_image]


with gr.Blocks(title="AI Art Director : Agent Workflow") as demo:
    gr.Markdown(
        "#  AI Art Director : Agent Workflow\n"
        "Upload an image and describe the vibe you want. "
        "AI agents will propose, apply, and critique edits to match your vision."
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            prompt_input = gr.Textbox(
                label="Describe the vibe you want", placeholder="e.g. dreamy, vintage, vibrant..."
            )
            submit_btn = gr.Button("Go!")
        with gr.Column():
            gallery = gr.Gallery(label="Workflow Images", show_label=True)
            critiques = gr.HighlightedText(label="Agent Comments", show_label=True)
            final = gr.Image(label="Final Image", show_label=True)

    submit_btn.click(
        process_image_with_agents,
        inputs=[image_input, prompt_input],
        outputs=[gallery, critiques, final],
    )

if __name__ == "__main__":
    demo.launch()
