import argparse
import base64
import os
import tempfile

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from PIL import Image
from smolagents import CodeAgent, InferenceClientModel

import filters as flt
import judges as jdg

LANGFUSE_PUBLIC_KEY = os.environ["LANGFURE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_AUTH = base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"  # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


HUGGING_FACE_TOKEN = os.environ["HUGGING_FACE_TOKEN"]

image_operator_model = InferenceClientModel(
    model_id="Qwen/Qwen3-32B", provider="nebius", token=HUGGING_FACE_TOKEN, max_tokens=5000
)

picture_operator_prompt = """
    You are an image processing agent capable of applying visual enhancements to images.
    Your task is to process images based on directives from an art director.
    You can adjust the following parameters: contrast, exposure, saturation,
    shadows/highlights, temperature, tint, hue, saturation (per color), and luminance (per color).

    Inputs:
    Original image path
    Output image path
    User query (serves as creative reference)
    Ordered list of enhancements to apply

    Rules:
    Always begin with the original image for each set of transformations.
    Never reuse previously processed images as a starting point.
    Apply only the operations explicitly listed. Do not invent or introduce new tools or methods.
    For each enhancement, you'll receive a qualitative assessment such as “too much,” “too little,” or “just right.”
    Use your understanding of image processing tools to translate qualitative feedback into quantitative adjustments.
    After completing the initial enhancement list, pass the resulting image to the critic for evaluation.
    Adjust parameters based on the critic’s feedback.
    Iterate until the critic responds with “just right” for all changes.
    Once all enhancements are satisfactory, your task is complete.
    You must call to conversion function rgb_to_hsl and hsl_to_rgb only once for each.
    You will be fined every time you exceed 1 call for each function.
    You must save the image only once.

    The code must be structured as follows :
    1. You load the image
    2. You call the needed that take a rgb image as input
    3. You convert the image into h, s, li canals
    4. You adjust the saturation, luminance and hue of color channels
    5. You convert back into r,g,b
    6. You save the rgb image

    Examples:
    Prompt: Increase contrast by a lot and orange saturation by a bit.
    Answer:
    img = load_image(path=image_path)

    # Apply strong contrast adjustment
    img = adjust_contrast(
        img=img,
        factor=1.1  # 1.1 is considered a lot per tool documentation
    )

    # Convert to HSL for color-specific adjustments
    h, s, li = rgb_to_hsl(img)

    # Enhance orange saturation slightly
    h, s, li = adjust_saturation_color(
        h=h,
        s=s,
        li=li,
        color='orange',
        factor=0.2)
    # Convert back to RGB

    # Save the result
    save_image(
        h, s, li
        output_directory=output_directory
    )
    critic(output_directory=output_path,
            original_image_path=image_path,
            user_prompt=user_query,
            list_of_enhancements=enhancements)



    Prompt 2: increase contrast by a lot, raise saturation medium,
    add some vignetage, a very little of grain,
    raise the exposition by a tiny bit,
    raise the orange saturation by a bit, the blue yellow and green luminance by a lot

    Answer:
    img = load_image(path=image_path)

    # Convert back to RGB and apply global adjustments
    img = adjust_contrast(img, factor=1.5)  # Increase contrast a lot
    img = adjust_saturation(img, factor=1.2)  # Medium saturation increase
    img = adjust_exposure(img, ev=0.05)  # Tiny exposure increase
    img = add_vignette(img, strength=0.5)  # Add some vignette
    img = add_grain(img, amount=0.01)  # Add very little grain

    h, s, li = rgb_to_hsl(img)
    # Adjust orange saturation by a bit
    h, s, li = adjust_saturation_color(h, s, li, color='orange', factor=1.1)

    # Increase luminance for blue, yellow, and green by a lot (factor=2)
    for color in ['blue', 'yellow', 'green']:
        h, s, li = adjust_luminance_color(h, s, li, color=color, factor=2)

    # Save the processed image
    save_image(h, s, li, output_directory=output_directory)

    # Final confirmation
    critic(output_directory=output_path,
            original_image_path=image_path,
            user_prompt=user_query,
            list_of_enhancements=enhancements)


    Prompt 3:
    Here’s my proposal to enhance the image with a vibrant,
    Instagram-style aesthetic while preserving its serene energy:
    **1.    Global Adjustments**
    **Contrast**: Slightly increased to add depth to the balloons'
    patterns without flattening the sky's gradient.
    **Exposure**: Brightened moderately to amplify the sunlit atmosphere,
    especially on the foreground balloon's geometric design.
    **Saturation**: Boosted a lot to intensify the mosaic of colors
    (red, orange, yellow, green, blue) on the balloons, making them feel more dynamic.
    **Temperature**: Warmed up to enhance the golden-hour glow, complementing the balloons' warm gradients.
    **Shadows/Highlights**: Shadows lifted slightly to reveal texture
    in the balloon fabrics, while highlights are tamed
    to avoid blowing out the sky's delicate clouds.
    **2. Color-Specific Tweaks**
    **Red**: Boosted saturation significantly for the background Red Bull
    balloon to make the brand text pop, while slightly increasing
    luminance to prevent it from feeling too heavy.
    **Orange**: Enhanced hue slightly, shifting toward amber to deepen the
    middle balloon's gradient, adding warmth without muddiness.
    **Blue**: Adjusted hue to a richer cobalt tone in the foreground balloon's
    pattern, making the geometric shapes stand out against warmer hues.
    **Green**: Increased luminance moderately in the foreground balloon's
    green sections to balance the vibrant reds and oranges.
    **3. Subtle Textures**
    **Vignette**: Applied barely, with a subtle darkening at the corners
    to frame the balloons without distracting from the sky's serenity.
    **Grain**: Omitted entirely—this scene’s tranquility works best with a clean, smooth finish.
    **Result**: A luminous, hyper-saturated scene where the balloons’ colors
    feel bolder and more immersive, the sky appears crisper,
    and the overall mood is elevated to evoke joyful adventure.
    The adjustments amplify the image’s natural vibrancy
    without sacrificing its peaceful essence.


    Answer:
    img = load_image(path=image_path)

    # Apply global adjustments in RGB space
    img = adjust_exposure(img=img,
    ev=0.1  # Moderate brightening for golden-hour amplification
    )

    img = adjust_contrast(
        img=img,
        factor=1.05  # Slightly increased depth for balloon patterns
    )

    img = adjust_saturation(img=img, factor=1.8  # "Boosted a lot" to enhance vibrancy
    )

    img = adjust_temperature(img=img, delta=500  # Warm up by 500 mireds for golden-hour glow
    )

    img = adjust_shadows_highlights(img=img, shadow=1.1,  # Slight shadow lifting to reveal textures
        highlight=0.9  # Tame highlights to preserve sky colors
    )

    img = add_vignette(
        img=img,
        strength=0.3  # Subtle corner darkening for framing effect
    )

    # Convert to HSL for color-specific adjustments
    h, s, li = rgb_to_hsl(img)

    # Red color tweaks: Boost saturation and luminance
    h, s, li = adjust_saturation_color(
        h=h,
        s=s,
        li=li,
        color='red',
        factor=1.4  # "Significantly" boosted saturation
    )
    h, s, li = adjust_luminance_color(
        h=h,
        s=s,
        li=li,
        color='red',
        factor=1.1  # Slight luminance lift to avoid heaviness
    )

    # Orange hue shift toward amber
    h, s, li = adjust_hue_color(
        h=h,
        s=s,
        li=li,
        color='orange',
        delta=15  # 15° shift = slight hue adjustment
    )

    # Blue hue adjustment to cobalt
    h, s, li = adjust_hue_color(
        h=h,
        s=s,
        li=li,
        color='blue',
        delta=15  # Slight shift to richer tones
    )

    # Green luminance increase for balance
    h, s, li = adjust_luminance_color(
        h=h,
        s=s,
        li=li,
        color='green',
        factor=1.2  # Moderate luminance lift for balance
    )

    # Save final enhanced image
    save_image(
        h=h, s=s, li=li,
        output_directory=output_path
    )

    critic(output_directory=output_path,
            original_image_path=image_path,
            user_prompt=user_query,
            list_of_enhancements=enhancements)
    """

picture_operator = CodeAgent(
    tools=[
        flt.adjust_contrast,
        flt.adjust_exposure,
        flt.adjust_saturation,
        flt.adjust_shadows_highlights,
        flt.adjust_hue_color,
        flt.adjust_saturation_color,
        flt.add_vignette,
        flt.add_grain,
        flt.save_image,
        flt.load_image,
        jdg.critic,
        flt.rgb_to_hsl,
    ],
    model=image_operator_model,
    name="PictureOperator",
    description=picture_operator_prompt,
    managed_agents=[],
    max_steps=4,
)


def resize_longest_side_to_500(image_path):
    """
    Resize the image so that its longest side is 500 pixels, preserving aspect ratio.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
    """
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_resized{ext}"
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
    return output_path


def run_photo_enchancement_agent(
    query: str,
    image_path: str = "small_test_image.jpg",
    output_directory: str | None = None,
):
    """
    Run the photo enhancement agent with the provided query and image path.

    Args:
        query (str): The user query for the agent.
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
    """

    # Create a temporary file for the resized image
    if not output_directory:
        output_directory = tempfile.mkdtemp()

    resized_image_path = resize_longest_side_to_500(image_path=image_path)
    image_path = resized_image_path
    directions = jdg.call_to_director(image_path, query)
    picture_operator.run(
        picture_operator_prompt + "\n\nuser_query : " + directions,
        additional_args={
            "image_path": image_path,
            "output_directory": output_directory,
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

    default_directory = tempfile.mkdtemp()
    if args.agent == "ops":
        picture_operator.run(
            args.query,
            additional_args={
                "image_path": args.image_path,
                "output_directory": default_directory,
            },
        )
    elif args.agent == "dir":
        directions = jdg.call_to_director(args.image_path, args.query)
        picture_operator.run(
            picture_operator_prompt + "start with those instructions :" + directions,
            additional_args={
                "image_path": args.image_path,
                "output_path": default_directory,
                "enhancements": directions,
                "user_query": args.query,
            },
        )
