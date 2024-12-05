from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ThumbnailResponse(BaseModel):
    """
    Comprehensive model for thumbnail generation response
    """
    id: int = Field(..., description="Unique identifier for the thumbnail")
    path: str = Field(..., description="File path of the generated thumbnail")
    emotions: Dict[str, Any] = Field(default_factory=dict, description="Detected emotions for the thumbnail")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the thumbnail"
    )
class EditThumbnailRequest(BaseModel):
    """
    Model for editing a thumbnail with creative options
    """
    thumbnail_path: str
    emoji: Optional[str] = None
    text: Optional[str] = None
    font_path: Optional[str] = Field(
        default=None, 
        description="Path to the font file for text overlay"
    )
    font_size: Optional[int] = Field(
        default=24, 
        description="Size of the font for text overlay"
    )
    brightness: Optional[float] = Field(
        default=None, 
        description="Brightness adjustment for the thumbnail",
        ge=-1.0,  # greater than or equal to -1
        le=1.0    # less than or equal to 1
    )

    class Config:
        """
        Provide creative examples for documentation and testing
        """
        schema_extra = {
            "example": {
                "thumbnail_path": "/videos/epic_moment.jpg",
                "emoji": "🚀🔥", # Multiple emojis for more impact
                "text": "Legendary Gameplay Highlight", # More dynamic text
                "font_path": "/fonts/gaming_font.ttf", # Themed font path
                "font_size": 48, # Larger, more prominent font
                "brightness": 0.1 # Slight brightness enhancement
            },
            "examples": [
                {
                    "thumbnail_path": "/videos/nature_scene.jpg",
                    "emoji": "🌿🌄", 
                    "text": "Breathtaking Wilderness Moment",
                    "font_path": "/fonts/nature_script.ttf",
                    "font_size": 36,
                    "brightness": 0.2
                },
                {
                    "thumbnail_path": "/videos/sports_clip.jpg",
                    "emoji": "⚽🏆", 
                    "text": "Winning Goal of the Season!",
                    "font_path": "/fonts/sports_bold.ttf",
                    "font_size": 42,
                    "brightness": 0.15
                },
                {
                    "thumbnail_path": "/videos/music_performance.jpg",
                    "emoji": "🎸🎤", 
                    "text": "Epic Musical Journey",
                    "font_path": "/fonts/rock_style.ttf",
                    "font_size": 40,
                    "brightness": 0.05
                },
                {
                    "thumbnail_path": "/videos/cooking_show.jpg",
                    "emoji": "👨‍🍳🍽️", 
                    "text": "Culinary Masterpiece Unveiled",
                    "font_path": "/fonts/chef_script.ttf",
                    "font_size": 34,
                    "brightness": 0.1
                }
            ]
        }

# Optional: If you want to provide preset themes
class ThumbnailTheme:
    """
    Predefined themes for thumbnail editing
    """
    GAMING = {
        "emoji": "🎮🚀",
        "text": "Epic Gaming Moment",
        "font_path": "/fonts/gaming_font.ttf",
        "font_size": 48,
        "brightness": 0.1
    }

    NATURE = {
        "emoji": "🌿🌄",
        "text": "Nature's Breathtaking Beauty",
        "font_path": "/fonts/nature_script.ttf",
        "font_size": 36,
        "brightness": 0.2
    }

    SPORTS = {
        "emoji": "⚽🏆",
        "text": "Moment of Triumph",
        "font_path": "/fonts/sports_bold.ttf",
        "font_size": 42,
        "brightness": 0.15
    }

    MUSIC = {
        "emoji": "🎸🎤",
        "text": "Musical Journey",
        "font_path": "/fonts/rock_style.ttf",
        "font_size": 40,
        "brightness": 0.05
    }