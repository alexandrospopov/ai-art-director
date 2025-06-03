from smolagents import CodeAgent, HfApiModel, tool
from filters import *
import os

HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

# Initialize the model and agent
model = HfApiModel(token=HUGGING_FACE_TOKEN)
agent = CodeAgent(tools=[adjust_contrast, load_image_as_bgr, save_image], model=model)

# Run the agent
agent.run("Adjust the contrast of the image in image_path by a factor of 1.5. Save the image to output_path.", 
          additional_args={'image_path': 'toa-heftiba-Xmn-QXsVL4k-unsplash.jpg',
                           "output_path": 'adjusted_image.jpg'})
