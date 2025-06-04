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
def propose_operations(image_path: str, user_prompt: str = "Improve this image.") -> str:
    """
    Analyzes the provided image and suggests a series of enhancement operations.

    Args:
        image_path (str): The file path to the image to be analyzed.
        user_prompt (str): Additional instructions or context provided by the user.

    Returns:
        str: A response from the AI art director suggesting operations to apply to the image.
    """

    system_prompt = (
        "You are an AI art director. Your task is to propose a sequence of image enhancement operations "
        "to transform the provided image according to the user's request. "
        "Consider the following possible operations:\n"
        "- adjust_contrast\n"
        "- adjust_exposure\n"
        "- adjust_saturation\n"
        "- adjust_shadows_highlights\n"
        "- adjust_temperature\n"
        "- adjust_tint\n"
        "- adjust_hue_color\n"
        "- adjust_saturation_color\n"
        "- adjust_luminance_color\n"
        "- add_vignette\n"
        "- denoise_image\n"
        "- add_grain\n"
        "In particular, you should use the methods that adjust colors luminance staturation and hue."
        "Here's an example of how to use these methods:\n"
        "1. Boost foliage without oversaturating skin"
        r"img = adjust_saturation_color(img, 'green', 1.2)  # +20\% green saturation"
        "img = adjust_luminance_color(img, 'green', 1.1)   # brighten greens a bit\n"
        "2. Warm skin tones without touching the whole image"
        "img = adjust_hue_color(img, 'orange', -5)         # shift orange hue toward red"
        "img = adjust_luminance_color(img, 'orange', 1.05) # subtle lift in skin brightness\n"
        "3. Darken blue sky for drama (a common Lightroom trick)"
        "img = adjust_luminance_color(img, 'blue', 0.85)   # darken blues"
        "img = adjust_saturation_color(img, 'blue', 1.15)  # intensify sky color\n\n"
        "For each operation you suggest, specify the amount to apply in relative units (e.g., +10%). "
        "Do not perform any operations or evaluate the image yourself. "
        "Only suggest a list of operations and their recommended amounts. "
        "Evaluation will be handled by a separate critic."
    )
    response = call_to_llm(
        image_path, model="Qwen/Qwen2.5-VL-72B-Instruct", system_prompt=system_prompt, user_prompt=user_prompt
    )
    return response["choices"][0]["message"]["content"]


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
        "You have 3 options, either the changes are bad and should be reverted, "
        "or the changes are on the right track but need further adjustments, "
        "or the changes are good and should be kept and the image saved."
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
