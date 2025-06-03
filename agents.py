import argparse
import os
import tempfile

from smolagents import CodeAgent, InferenceClientModel

import filters as flt

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

model = InferenceClientModel(token=HUGGING_FACE_TOKEN)
picture_operator = CodeAgent(
    tools=[flt.adjust_contrast, flt.load_image_as_bgr, flt.save_image],
    model=model,
    name="PictureOperator",
    description=(
        "Performs operations on images, such as adjusting contrast, loading images, and saving them. "
        "Give it your query as an argument, as well as the path to the image and the output path."
    ),
)

art_director = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[picture_operator],
    name="ArtDirector",
    description=(
        "Deciess which filter to apply to an image. "
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
            additional_args={
                "image_path": args.image_path,
            },
        )
