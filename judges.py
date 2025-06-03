import base64
import io
import os

from openai import OpenAI
from PIL import Image
from smolagents import tool


def pil_image_to_data_url(pil_image, format="JPEG"):
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    mime_type = f"image/{format.lower()}"
    base64_encoded_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"


def call_to_llm(image_path, model, system_prompt=None, user_prompt=None):
    img = Image.open(image_path)
    data_url = pil_image_to_data_url(img, format=img.format)

    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_TOKEN"),
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.6,
    )

    return completion.to_dict()


@tool
def propose_operations(image_path: str, user_prompt: str = "Improve this image.") -> dict:
    """
    Analyzes the provided image and suggests a series of enhancement operations.

    Args:
        image_path (str): The file path to the image to be analyzed.
        user_prompt (str): Additional instructions or context provided by the user.

    Returns:
        dict: The response from the language model containing a list of 5 different combinations
        of image enhancement operations, such as applying filters, cropping, or other adjustments.
    """

    system_prompt = (
        "You are an AI art director. "
        "Your task is to analyze the provided image and suggest a series of operations to make"
        "the user happy. You must suggest applying filters, cropping, or other adjustments. "
        "Provide a list of 5 different combinations of operations without explanations."
        "No introductions, no conclusions, just the list of operations."
    )
    response = call_to_llm(
        image_path, model="google/gemma-3-27b-it", system_prompt=system_prompt, user_prompt=user_prompt
    )
    return response


if __name__ == "__main__":
    res = propose_operations(image_path="small_test_image.jpg")
    print(res["choices"][0]["message"]["content"])
