import base64
import io
import os
import tempfile

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
        "You take strong decisions with important consequences on the image.\n"
        "Consider the following possible operations:\n"
        "- adjust_contrast\n"
        "- adjust_exposure\n"
        "- adjust_saturation\n"
        "- adjust_shadows_highlights\n"
        "- adjust_temperature\n"
        "- adjust_tint\n"
        "- adjust_hue_color colors are [red, orange, yellow, green, aqua, blue, purple, magenta]\n"
        "- adjust_saturation_color colors are [red, orange, yellow, green, aqua, blue, purple, magenta]\n"
        "- adjust_luminance_color colors are [red, orange, yellow, green, aqua, blue, purple, magenta]\n"
        "- add_vignette\n"
        "- add_grain\n"
        "In particular, you should use the methods that adjust colors luminance staturation and hue. "
        "I want at least 3 colors to be adjusted.\n"
        "When citing the methods, describe qualitatively how much the effect should be applied : "
        "a lot, bearly, to the maximum, ..."
    )
    response = call_to_llm(
        image_path, model="Qwen/Qwen2.5-VL-72B-Instruct", system_prompt=system_prompt, user_prompt=user_prompt
    )
    return response["choices"][0]["message"]["content"]


def concatenate_images_side_by_side(image_path1: str, image_path2: str) -> str:
    """
    Concatenates two images side by side, saves the result to a temporary file, and returns the file path.

    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.

    Returns:
        str: Path to the concatenated image saved in a temporary file.
    """
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    separator_width = 5
    total_width = img1.width + separator_width + img2.width
    max_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))
    new_img.paste(img1, (0, 0))
    # Draw the black separator (already black background, but for clarity)
    separator_box = (img1.width, 0, img1.width + separator_width, max_height)
    new_img.paste(Image.new("RGB", (separator_width, max_height), (0, 0, 0)), separator_box)
    new_img.paste(img2, (img1.width + separator_width, 0))

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    new_img.save(temp_file, format="JPEG")
    temp_file.close()
    return temp_file.name


@tool
def critic(new_image_path: str, original_image_path: str, user_prompt: str, list_of_enhancements: str) -> str:
    """
    Evaluates the new image against the old image and provides feedback.

    Args:
        new_image_path (str): The file path to the new image.
        original_image_path (str): The file path to the new image.
        user_prompt (str): Additional instructions or context provided by the user.
        list_of_enhancements (str): the list of of all enhancements applied to the image.

    Returns:
        str: Feedback on the changes made to the image.
    """
    print("list_of_enhancements: ", list_of_enhancements)
    path_to_concat = concatenate_images_side_by_side(original_image_path, new_image_path)

    # iterate each time between the ops and the critic

    system_prompt = (
        "You are an AI art critic. "
        "You will be provided a single image, consisting of 2 images side by side.\n"
        "On the left side, you will see the original image, "
        "and on the right side, you will see the new image enhanced by AI.\n"
        "Your task is to evaluate the changes made to image 2.\n"
        "You will be provided with a prompt that describes the user's request to improve the image.\n"
        "Does the image 2 respect the desire of the user ?\n"
        "The application of the filters must be noticeable."
        "The minimal increment is 10%. Don't be too subtle.\n"
        "You can refine the list of the modifications to apply to the image.\n"
        "You must not invent new methods or tools, only use the ones provided.\n"
        "You must include in your answer a rate out of 10 of the image."
        "If you rate the rate above 8/10, just accept the image as it is."
    )

    response = call_to_llm(
        path_to_concat,
        model="Qwen/Qwen2.5-VL-72B-Instruct",
        system_prompt=system_prompt,
        user_prompt=f"the user wishes for : {user_prompt}.\n The enhancement applied are {list_of_enhancements}",
    )

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    res = propose_operations(image_path="small_test_image.jpg")
    print(res["choices"][0]["message"]["content"])
