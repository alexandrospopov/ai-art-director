import os

from smolagents import CodeAgent, HfApiModel

import filters as flt

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

# Initialize the model and agent
model = HfApiModel(token=HUGGING_FACE_TOKEN)
picture_operator = CodeAgent(
    tools=[flt.adjust_contrast, flt.load_image_as_bgr, flt.save_image],
    model=model,
    name="Picture Operator",
    description=(
        "Performs operations on images, such as adjusting contrast, loading images, and saving them. "
        "Give it your query as an argument, as well as the path to the image and the output path."
    ),
)

# Run the agent
picture_operator.run(
    "Adjust the contrast of the image in image_path by a factor of 1.5. Save the image to output_path.",
    additional_args={
        "image_path": "toa-heftiba-Xmn-QXsVL4k-unsplash.jpg",
        "output_path": "adjusted_image.jpg",
    },
)
