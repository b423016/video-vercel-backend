from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import emoji
import os


class ImageEditor:
    def __init__(self):
        # Default font and emoji directory
        self.default_font_path = os.path.join('assets', 'fonts', 'Arial.ttf')
        self.emoji_dir = os.path.join('assets', 'emojis')

    def edit_thumbnail(self,
                       thumbnail_path: str,
                       emoji: str = None,
                       text: str = None,
                       font_path: str = None,
                       font_size: int = 36,
                       brightness_adjustment: float = 0):
        """
        Edit thumbnail with emoji, text, and brightness adjustment
        """
        # Open the image
        image = Image.open(thumbnail_path)

        # Adjust brightness
        if brightness_adjustment != 0:
            image = self._adjust_brightness(image, brightness_adjustment)

        # Add emoji if specified
        if emoji:
            image = self._add_emoji(image, emoji)

        # Add text if specified
        if text:
            image = self._add_text(
                image,
                text,
                font_path or self.default_font_path,
                font_size
            )

        return image

    def _adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """
        Adjust image brightness
        factor > 1 increases brightness
        factor < 1 decreases brightness
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _add_emoji(self, image: Image.Image, emoji_name: str) -> Image.Image:
        """
        Add emoji to the image
        """
        try:
            # Convert emoji to filename
            emoji_filename = f"{emoji_name}.png"
            emoji_path = os.path.join(self.emoji_dir, emoji_filename)

            # Open emoji image
            emoji_img = Image.open(emoji_path).convert("RGBA")

            # Resize emoji (10% of image width)
            emoji_size = int(image.width * 0.1)
            emoji_img = emoji_img.resize((emoji_size, emoji_size))

            # Paste emoji on top right corner
            image.paste(emoji_img,
                        (image.width - emoji_size - 10, 10),
                        emoji_img)
        except Exception as e:
            print(f"Error adding emoji: {e}")

        return image

    def _add_text(self,
                  image: Image.Image,
                  text: str,
                  font_path: str,
                  font_size: int) -> Image.Image:
        """
        Add text to the image
        """
        draw = ImageDraw.Draw(image)

        # Load font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()

        # Calculate text position (bottom center)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        position = (
            (image.width - text_width) // 2,
            image.height - text_height - 20
        )

        # Add text with shadow for better visibility
        draw.text((position[0] + 2, position[1] + 2), text, font=font, fill=(0, 0, 0, 128))
        draw.text(position, text, font=font, fill=(255, 255, 255, 255))

        return image