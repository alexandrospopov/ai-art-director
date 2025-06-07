import argparse
import os
import tempfile

from PIL import Image
from smolagents import CodeAgent, InferenceClientModel

import filters as flt
import judges as jdg

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

image_operator_model = InferenceClientModel(
    model_id="Qwen/Qwen3-30B-A3B",
    provider="nebius",
    token=HUGGING_FACE_TOKEN,
    max_tokens=5000,
)

picture_operator_prompt = (
    "You are an image processing agent, and you can perform operations on images, such as adjusting contrast, "
    "exposure, saturation, shadows/highlights, temperature, tint, hue/color, saturation/color, luminance/color."
    "You take as input the path to the original image, the path to the output image, "
    "the user query, that will serve as a reference, and a list of enhancements to apply to the image."
    "You must always use the original image when you start a set of transformations."
    "DO NOT USE previously created images."
    "This list comes from an art director. You must apply the operations in the list "
    "For each operation, you will receive a qualitative estimation of the change, "
    "like 'too much', 'too little', or 'just right'."
    "Use your knowledge of the tools to adjuste the parameters."
    "Execute only the operations that are proposed as tools. Do not invent new methods or tools."
    "After applying the operations, you must pass the resulting image to the critic for evaluation."
    "The critic will provide feedback on whether the change is too much, too little, or just right."
    "This will help you find just the right variable for each operation."
    "If the score the critic gives is higher than 7, you can save the image."
)

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
)


def resize_longest_side_to_500(image_path, output_path):
    """
    Resize the image so that its longest side is 500 pixels, preserving aspect ratio.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
    """
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


def run_photo_enchancement_agent(
    query: str,
    image_path: str = "small_test_image.jpg",
    output_path: str = "output.jpg",
):
    """
    Run the photo enhancement agent with the provided query and image path.

    Args:
        query (str): The user query for the agent.
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
    """

    # Create a temporary file for the resized image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        resized_image_path = tmp.name
    resize_longest_side_to_500(image_path=image_path, output_path=resized_image_path)
    image_path = resized_image_path
    directions = jdg.call_to_director(image_path, query)
    picture_operator.run(
        picture_operator_prompt + "\n\nuser_query : " + directions,
        additional_args={
            "image_path": image_path,
            "output_path": output_path,
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
    default_output_path = os.path.join(tempfile.mkdtemp(), "output.jpg")
    if args.agent == "ops":
        picture_operator.run(
            args.query,
            additional_args={
                "image_path": args.image_path,
                "output_path": default_output_path,
            },
        )
    elif args.agent == "dir":
        directions = jdg.call_to_director(args.image_path, args.query)
        picture_operator.run(
            picture_operator_prompt + "start with those instructions :" + directions,
            additional_args={
                "image_path": args.image_path,
                "output_path": default_output_path,
                "enhancements": directions,
                "user_query": args.query,
            },
        )
