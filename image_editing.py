import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from typing import Optional, Tuple
class ImageEditor:
    def __init__(self):
        # Use default system font for simplicity
        self.default_font = ImageFont.load_default()

    def edit_thumbnail(self, thumbnail_path: str, text: Optional[str] = None,
                       font_size: int = None, text_color: Tuple[int, int, int] = (255, 255, 255),
                       brightness_adjustment: float = 0, upscale: bool = True) -> Image.Image:
        # Open the thumbnail
        image = Image.open(thumbnail_path)

        # Adjust brightness
        if brightness_adjustment != 0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1 + brightness_adjustment)

        # Add text if provided
        if text:
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arial.ttf", font_size)
            text_position = (10, 10)  # Default position (top-left corner)
            draw.text(text_position, text, fill=text_color, font=font)

        # Optionally upscale the image
        if upscale:
            image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)

        return image

    def _upscale_image(self, image: Image.Image, target_resolution: tuple[int, int] = (1920, 1080)) -> Image.Image:
        """
        Upscale image to 1080p, cropping if necessary to maintain aspect ratio.
        """
        original_width, original_height = image.size
        target_width, target_height = target_resolution

        # Calculate aspect ratios
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height

        if original_aspect > target_aspect:
            # Image is wider than 16:9, crop width
            new_width = int(original_height * target_aspect)
            left = (original_width - new_width) // 2
            right = left + new_width
            image = image.crop((left, 0, right, original_height))
        elif original_aspect < target_aspect:
            # Image is taller than 16:9, crop height
            new_height = int(original_width / target_aspect)
            top = (original_height - new_height) // 2
            bottom = top + new_height
            image = image.crop((0, top, original_width, bottom))

        # Resize to exactly 1080p
        return image.resize(target_resolution, Image.Resampling.LANCZOS)

    def _adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """
        Adjust image brightness.
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _add_emoji(self, image: Image.Image, emoji_path: str, position: tuple) -> Image.Image:
        """
        Add an emoji to the image at a specified position.
        """
        try:
            emoji_img = Image.open(emoji_path).convert("RGBA")
            emoji_size = int(image.width * 0.1)
            emoji_img = emoji_img.resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)

            if not position:
                position = (image.width - emoji_size - 10, 10)

            image.paste(emoji_img, position, emoji_img)
        except Exception as e:
            raise RuntimeError(f"Error adding emoji: {e}")

        return image

    def _add_text(self, image: Image.Image, text: str, position: tuple, color: tuple, font_size: int) -> Image.Image:
        """
        Add text to the image at a specified position.
        """
        draw = ImageDraw.Draw(image)
        font = self.default_font

        # Calculate text size
        text_width, text_height = draw.textsize(text, font=font)

        # Validate position
        if not position:
            position = (
                (image.width - text_width) // 2,
                image.height - text_height - 20
            )

        # Add text
        draw.text(position, text, font=font, fill=color)
        return image
