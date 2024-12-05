import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import shutil

from app.video_processing import VideoProcessor
from app.image_editing import ImageEditor
from app.models import ThumbnailResponse, EditThumbnailRequest

app = FastAPI(title="Thumbnail Generator")

# Ensure directories exist
os.makedirs('videos', exist_ok=True)
os.makedirs('thumbnails', exist_ok=True)

video_processor = VideoProcessor()
image_editor = ImageEditor()


@app.post("/generate_thumbnails/")
async def generate_thumbnails(file: UploadFile = File(...)):
    """
    Generate 5 random thumbnails from an uploaded video
    """
    # Save uploaded video
    video_path = os.path.join('videos', f"{uuid.uuid4()}_{file.filename}")
    with open(video_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Generate thumbnails
        thumbnails = video_processor.extract_thumbnails(video_path)

        # Save generated thumbnails
        thumbnail_paths = []
        for i, thumbnail in enumerate(thumbnails):
            thumbnail_path = os.path.join('thumbnails', f'thumbnail_{i + 1}.jpg')
            thumbnail.save(thumbnail_path)
            thumbnail_paths.append(thumbnail_path)

        return JSONResponse(content={
            "thumbnails": [
                {
                    "id": i + 1,
                    "path": path,
                    "emotions": video_processor.detect_frame_emotions(path)
                }
                for i, path in enumerate(thumbnail_paths)
            ]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit_thumbnail/")
async def edit_thumbnail(request: EditThumbnailRequest):
    """
    Edit a specific thumbnail
    """
    try:
        edited_image = image_editor.edit_thumbnail(
            request.thumbnail_path,
            emoji=request.emoji,
            text=request.text,
            font_path=request.font_path,
            font_size=request.font_size,
            brightness_adjustment=request.brightness
        )

        # Save edited image
        edited_path = os.path.join('thumbnails', f'edited_{uuid.uuid4()}.jpg')
        edited_image.save(edited_path)

        return {"edited_thumbnail_path": edited_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    """
    Serve a specific thumbnail
    """
    thumbnail_path = os.path.join('thumbnails', filename)
    if not os.path.exists(thumbnail_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumbnail_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)