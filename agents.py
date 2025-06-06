import argparse
import os
import tempfile

from smolagents import CodeAgent, InferenceClientModel

import filters as flt
import judges as jdg

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

image_operator_model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    provider="nebius",
    token=HUGGING_FACE_TOKEN,
    max_tokens=5000,
)

picture_operator_prompt = (
    "You are a picture operator, and you can perform operations on images, such as adjusting contrast, "
    "loading images, and saving them. Give it your query as an argument, as well as the path to the image"
    " and the output path."
    "Execute only the operations that are proposed as tools. Do not invent new methods or tools."
    "For each operation, you will receive a percentage change, like +10% or -5%."
    "After applying the operations, you must pass the resulting image to the critic for evaluation."
    "The critic will provide feedback on whether the change is too much, too little, or just right."
    "You can call the critic only 2 times per image."
    "You must adjust the operations based on the critic's feedback."
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
        flt.denoise_image,
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

art_director_model = InferenceClientModel(
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    provider="nebius",
    token=HUGGING_FACE_TOKEN,
)

art_director_prompt = (
    "You manage the relations between the art director, the picture operator and the critic."
    "You must present the images to improve to the art director, who will propose operations to apply to the image."
    "The quantitative parameters in operations are hard to predict, so the art director will propose operations "
    "with a percentage change, like +10% or -5%."
    "You must pass those operations to the picture operator, who will apply them to the image "
    "and refine the percentage changes with the help of the critic."
    "Once all operations are applied, and the critic is happy, you can save the final image."
    "You must not evaluate the image yourself, only pass it to the critic."
    "You must not propose operations yourself, only pass them to the picture operator."
    "You must not invent new methods or tools, only use the ones provided."
)
art_director = CodeAgent(
    tools=[jdg.propose_operations, jdg.critic],
    model=art_director_model,
    managed_agents=[picture_operator],
    description=art_director_prompt,
    name="Manager",
)


if __name__ == "__main__":
    # Run the agent
    parser = argparse.ArgumentParser(description="Run image processing agents.")
    parser.add_argument(
        "--agent",
        "-a",
        choices=["ops", "ops1", "dir"],
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
    elif args.agent == "ops1":
        directions = jdg.propose_operations(args.image_path, args.query)
        picture_operator.run(
            picture_operator_prompt + "\n\nuser_query : " + directions,
            additional_args={
                "image_path": args.image_path,
                "output_path": default_output_path,
            },
        )
    elif args.agent == "dir":
        art_director.run(
            art_director_prompt + "\n\nuser_query : " + args.query,
            additional_args={
                "image_path": args.image_path,
                "output_path": default_output_path,
            },
        )
