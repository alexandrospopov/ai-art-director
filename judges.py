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


def call_to_llm(image_path, model, system_prompt=None, user_prompt=None, second_image_path=None):
    img = Image.open(image_path)
    data_url = pil_image_to_data_url(img, format=img.format)

    user_content = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    if second_image_path:
        img2 = Image.open(second_image_path)
        data_url2 = pil_image_to_data_url(img2, format=img2.format)
        user_content.append({"type": "image_url", "image_url": {"url": data_url2}})

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
                "content": user_content,
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
        "Your task is to analyze the provided image and suggest one operation to make"
        "the user happy. You must suggest applying filters from the following list: "
        "adjust_contrast"
        "adjust_exposure"
        "adjust_saturation"
        "adjust_shadows_highlights"
        "adjust_temperature"
        "adjust_tint"
        "adjust_hue_color"
        "adjust_saturation_color"
        "adjust_luminance_color"
        "adjust_hsl_channel"
        "add_vignette"
        "denoise_image"
        "add_grain"
        r"In addition, you must suggest the amount of the operation to apply, in relative units : +10\% for instance."
        "You must suggest only one operation at a time, and you must not invent new methods or tools."
        "You must not perform any operations on the image yourself, only pass them to the picture operator."
        "You must not evaluate the image yourself, only pass it to the critic."
    )
    response = call_to_llm(
        image_path, model="google/gemma-3-27b-it", system_prompt=system_prompt, user_prompt=user_prompt
    )
    return response


@tool
def critic(new_image_path: str, old_image_path: str) -> str:
    """
    Evaluates the new image against the old image and provides feedback.

    Args:
        new_image_path (str): The file path to the new image.
        old_image_path (str): The file path to the old image.

    Returns:
        str: Feedback on the changes made to the image.
    """
    system_prompt = (
        "You are an AI art critic. "
        "Your task is to evaluate the changes made to an image. "
        "The first image in the new one and the second image is the old one. "
        "Compare the new image with the old one and provide feedback on the changes."
        "Your answer should be either 'good' or 'bad', "
        "indicating whether the changes made to the image are satisfactory or not."
    )
    user_prompt = "Evaluate the changes made to this image."

    response = call_to_llm(
        new_image_path,
        model="google/gemma-3-27b-it",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        second_image_path=old_image_path,
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    res = propose_operations(image_path="small_test_image.jpg")
    print(res["choices"][0]["message"]["content"])
