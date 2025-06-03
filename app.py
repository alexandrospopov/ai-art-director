import gradio as gr
from filters import apply_filters
from evaluators import evaluate_filters

def process_image(image):
    filtered_images = apply_filters(image)
    best_index, reasons = evaluate_filters(filtered_images)
    return filtered_images, f"Winner: Filter {best_index + 1} — {reasons[best_index]}"

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Gallery(label="Filtered Options"),
        gr.Textbox(label="Critique Agent's Verdict")
    ],
    title="🧠 AI Art Director – Filter Showdown",
    description="Upload an image and let AI agents apply, evaluate, and pick the best filter."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)