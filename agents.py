import argparse
import base64
import os
import tempfile

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from PIL import Image
from smolagents import CodeAgent, InferenceClientModel

import filters as flt
import judges as jdg

LANGFUSE_PUBLIC_KEY = os.environ["LANGFURE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"  # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

image_operator_model = InferenceClientModel(
    model_id="Qwen/Qwen3-32B", provider="nebius", token=HUGGING_FACE_TOKEN, max_tokens=5000
)

picture_operator_prompt = """
    You are an image processing agent capable of applying visual enhancements to images.
    Your task is to process images based on directives from an art director.
    You can adjust the following parameters: contrast, exposure, saturation,
    shadows/highlights, temperature, tint, hue, saturation (per color), and luminance (per color).

    Inputs:
    Original image path
    Output image path
    User query (serves as creative reference)
    Ordered list of enhancements to apply

    Rules:
    Always begin with the original image for each set of transformations.
    Never reuse previously processed images as a starting point.
    Apply only the operations explicitly listed. Do not invent or introduce new tools or methods.
    For each enhancement, you'll receive a qualitative assessment such as “too much,” “too little,” or “just right.”
    Use your understanding of image processing tools to translate qualitative feedback into quantitative adjustments.
    After completing the initial enhancement list, pass the resulting image to the critic for evaluation.
    Adjust parameters based on the critic’s feedback.
    Iterate until the critic responds with “just right” for all changes.
    Once all enhancements are satisfactory, your task is complete.
    """

picture_operator = CodeAgent(
    tools=[
        flt.adjust_contrast,
        flt.adjust_exposure,
        flt.adjust_saturation,
        flt.adjust_shadows_highlights,
        flt.adjust_temperature,
        flt.adjust_tint,
        flt.adjust_hue_color,
        flt.adjust_saturation_color,
        flt.adjust_luminance_color,
        flt.add_vignette,
        flt.add_grain,
        flt.save_image,
        flt.load_image,
        jdg.critic,
    ],
    model=image_operator_model,
    name="PictureOperator",
    description=picture_operator_prompt,
    managed_agents=[],
    max_steps=7,
)


def resize_longest_side_to_500(image_path):
    """
    Resize the image so that its longest side is 500 pixels, preserving aspect ratio.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
    """
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_resized{ext}"
    with Image.open(image_path) as img:
        width, height = img.size
        if width >= height:
            new_width = 500
            new_height = int((500 / width) * height)
        else:
            new_height = 500
            new_width = int((500 / height) * width)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(output_path)
    return output_path


def run_photo_enchancement_agent(
    query: str,
    image_path: str = "small_test_image.jpg",
    output_directory: str | None = None,
):
    """
    Run the photo enhancement agent with the provided query and image path.

    Args:
        query (str): The user query for the agent.
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
    """

    # Create a temporary file for the resized image
    if not output_directory:
        output_directory = tempfile.mkdtemp()

    resized_image_path = resize_longest_side_to_500(image_path=image_path)
    image_path = resized_image_path
    directions = jdg.call_to_director(image_path, query)
    picture_operator.run(
        picture_operator_prompt + "\n\nuser_query : " + directions,
        additional_args={
            "image_path": image_path,
            "output_directory": output_directory,
        },
    )


if __name__ == "__main__":
    # Run the agent
    parser = argparse.ArgumentParser(description="Run image processing agents.")
    parser.add_argument(
        "--agent",
        "-a",
        choices=["ops", "dir"],
        required=True,
        help="Which agent to run.",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Query to pass to the agent.",
    )
    parser.add_argument(
        "--image_path",
        "-i",
        type=str,
        required=False,
        default="small_test_image.jpg",
        help="Path to the input image.",
    )

    args = parser.parse_args()

    default_directory = tempfile.mkdtemp()
    if args.agent == "ops":
        picture_operator.run(
            args.query,
            additional_args={
                "image_path": args.image_path,
                "output_directory": default_directory,
            },
        )
    elif args.agent == "dir":
        directions = jdg.call_to_director(args.image_path, args.query)
        picture_operator.run(
            picture_operator_prompt + "start with those instructions :" + directions,
            additional_args={
                "image_path": args.image_path,
                "output_path": default_directory,
                "enhancements": directions,
                "user_query": args.query,
            },
        )
