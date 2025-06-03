from PIL import ImageEnhance

def apply_filters(image):
    filters = []

    # Filter 1: Brightness boost
    filters.append(ImageEnhance.Brightness(image).enhance(1.3))

    # Filter 2: Contrast boost
    filters.append(ImageEnhance.Contrast(image).enhance(1.5))

    # Filter 3: Color boost
    filters.append(ImageEnhance.Color(image).enhance(1.8))

    return filters