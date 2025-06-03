import argparse
import os
import tempfile

from PIL import Image
from smolagents import CodeAgent, InferenceClientModel

import filters as flt

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

image_operator_model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    provider="nebius",
    token=HUGGING_FACE_TOKEN,
    max_tokens=5000,
)

picture_operator = CodeAgent(
    tools=[
        flt.adjust_contrast,
        flt.load_image_as_bgr,
        flt.save_image,
        flt.adjust_saturation,
        flt.adjust_exposure,
        flt.denoise_image,
        flt.crop_image,
        flt.apply_vignette,
    ],
    model=image_operator_model,
    name="PictureOperator",
    description=(
        "Performs operations on images, such as adjusting contrast, loading images, and saving them. "
        "Give it your query as an argument, as well as the path to the image and the output path."
    ),
)

art_director_model = InferenceClientModel(
    model_id="Qwen/Qwen2-VL-72B-Instruct",
    provider="nebius",
    token=HUGGING_FACE_TOKEN,
    max_tokens=5000,
    flatten_messages_as_text=False,
)
art_director = CodeAgent(
    tools=[],
    model=art_director_model,
    managed_agents=[picture_operator],
    name="ArtDirector",
    description=(
        "Decides which filter to apply to an image. "
        "Give it your query as an argument, as well as the path to the image."
    ),
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
        default="test_image.jpg",
        help="Path to the input image.",
    )

    args = parser.parse_args()
    if args.agent == "ops":
        default_output_path = os.path.join(tempfile.mkdtemp(), "output.jpg")
        picture_operator.run(
            args.query,
            additional_args={
                "image_path": args.image_path,
                "output_path": default_output_path,
            },
        )
    elif args.agent == "dir":
        art_director.run(
            args.query,
            images=[Image.open(args.image_path)],
        )
