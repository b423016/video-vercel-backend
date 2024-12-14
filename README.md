# Thumbnail Generator

## Overview
An advanced AI-powered thumbnail generator using FastAPI that extracts key frames, detects scenes and emotions, and allows dynamic thumbnail editing.

## Features
- Video upload and thumbnail extraction
- This version processing is a bit slow I am working on it please be patient 

## UPDATES
- added the text size function
- added upscale to increase quality of image
- @todo will add a emoji pallete 
- will use ai for text positioning 
- give user an option to use ai for text generation(optional user input will be there)
- brightness is added

##Frontend 
- Take video path as input in string\
- then Backend return file path and zip file as json
- in the edit thumbnail
-  below is th eparam to be taken input as edit thumbnail frontend gives json in the below format
-   thumbnail_path: str
  
    text: Optional[str] = None
    
    font_size: Optional[int] = 40
    
    text_color: Optional[Tuple[int, int, int]] = (255, 255, 255)
    
    brightness_adjustment: Optional[float] = 0
    
    upscale: Optional[bool] = True
    
- Gives img path in return
    
