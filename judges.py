import base64
import io
import os
import re
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


def call_to_llm(model, image_path=None, system_prompt=None, user_prompt=None):

    user_content = []
    if image_path:
        img = Image.open(image_path)
        data_url = pil_image_to_data_url(img, format=img.format)
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})

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


def call_to_director(image_path: str, user_prompt):
    describe_prompt = "describe this image. Include color distribution and exposition description"
    image_description = call_to_llm(
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        image_path=image_path,
        user_prompt=describe_prompt,
        system_prompt="you are a helpful AI",
    )

    image_description = image_description["choices"][0]["message"]["content"]
    system_prompt = (
        "You are an AI art director. Your task is to propose a sequence of image enhancement operations "
        "Overall, the modifications you are making aim at mimicking the instagm filters effect."
        "You like to enhance pictures to make them full of light and color, happy and vibrant."
        "The vignette effet and the grain should be use be scarcity, only if makes a lot of sense."
        "Be careful to not overdue the shadows as well."
        "to transform the provided image according to the user's request. "
        "Your personnal bias is that you LOVE bright, blue skies. So whenever the sky plays an "
        "important role in the composition, make sure to give it a proper kick."
        "You take strong decisions with important consequences on the image.\n"
        "Consider the following possible operations:\n"
        "- adjust_contrast\n"
        "- adjust_exposure\n"
        "- adjust_saturation\n"
        "- adjust_shadows_highlights\n"
        "- adjust_hue_color colors are [red, orange, yellow, green, aqua, blue, purple, magenta]\n"
        "- adjust_saturation_color colors are [red, orange, yellow, green, aqua, blue, purple, magenta]'n"
        "- add_vignette\n"
        "- add_grain\n"
        "In particular, you should use the methods that adjust colors luminance staturation and hue. "
        "I want at least 3 colors to be adjusted.\n"
        "When citing the methods, you should say more or less and describe qualitatively"
        " how much the effect should be applied : a lot, bearly, to the maximum, ..."
        "DO NOT INCLUDE ANY NUMBER. SIMPLY DESCRIBE THE STRENGTH OF THE EFFECT."
    )
    print(image_description)
    directions = call_to_llm(
        "Qwen/Qwen3-235B-A22B",
        user_prompt=f"my image is : {image_description}. The user request is {user_prompt}",
        system_prompt=system_prompt,
    )
    directions_str = directions["choices"][0]["message"]["content"]
    if "<think>" in directions_str and "</think>" in directions_str:
        directions_str = re.sub(r"<think>.*?</think>", "", directions_str, flags=re.DOTALL)
    directions["choices"][0]["message"]["content"] = directions_str
    return directions["choices"][0]["message"]["content"]


@tool
def critic(output_directory: str, original_image_path: str, user_prompt: str, list_of_enhancements: str) -> str:
    """
    Evaluates the new image against the old image and provides feedback.

    Args:
        output_directory (str): Path to the output directory
        original_image_path (str): The file path to the new image.
        user_prompt (str): Additional instructions or context provided by the user.
        list_of_enhancements (str): the list of of all enhancements applied to the image.

    Returns:
        str: Feedback on the changes made to the image.
    """
    print("list_of_enhancements: ", list_of_enhancements)
    # Find all files matching the pattern "trial_i" where i is an integer and extension is .jpeg
    trial_files = [f for f in os.listdir(output_directory) if re.match(r"trial_\d+\.jpeg$", f)]
    # Sort files by descending integer i
    trial_files.sort(key=lambda x: (int(m.group(1)) if (m := re.search(r"trial_(\d+)\.jpeg$", x)) else -1))
    new_image_path = os.path.join(output_directory, trial_files[-1])
    path_to_concat = concatenate_images_side_by_side(original_image_path, new_image_path)

    # iterate each time between the ops and the critic

    system_prompt = (
        "You are an AI art critic. You provide a list adjustements to make the enhancements better\n"
        "You will be provided a single image, consisting of 2 images side by side."
        "On the left side, you will see the original image, "
        "and on the right side, you will see the new image enhanced by AI.\n"
        "Your task is to evaluate the changes made to image 2.\n"
        "You will be provided with a prompt that describes the user's request to improve the image.\n"
        "Does the image 2 respect the desire of the user ?\n"
        "The application of the filters must be noticeable, but the effect should not look surreal."
        "Above all, you dislike unnatural colors and color distorsions due to overloaded enchancements."
        "Overall, the modifications you are making aim at mimicking the instagm filters effect."
        "Unless, specifically asked the contrary, you should stick to clear pictures full of light. "
        "The vignette effet and the grain should be use be scarcity, only if makes a lot of sense."
        "Here are the possible operations:\n"
        "- adjust_contrast\n"
        "- adjust_exposure\n"
        "- adjust_saturation\n"
        "- adjust_shadows_highlights\n"
        "- adjust_temperature\n"
        "- adjust_tint\n"
        "- adjust_hue_color colors are [red, orange, yellow, green, aqua, blue, purple, magenta]\n"
        "- adjust_saturation_color colors are [red, orange, yellow, green, aqua, blue, purple, magenta]'n"
        "- add_vignette\n"
        "- add_grain\n"
        "FOR EACH, you must say what should be done to improve the image"
        "Tool: Contrast\n"
        "Suggestion: Reduce significantly the contrast\n"
        "Confidence: 0.9\n"
    )

    response = call_to_llm(
        "Qwen/Qwen2.5-VL-72B-Instruct",
        image_path=path_to_concat,
        system_prompt=system_prompt,
        user_prompt=f"the user wishes for : {user_prompt}.\n The enhancement applied are {list_of_enhancements}",
    )

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # res = propose_operations(image_path="small_test_image.jpg")
    # print(res["choices"][0]["message"]["content"])
    directions = call_to_director("small_forest.jpg", "give it a winter vibe")
    print(directions)
