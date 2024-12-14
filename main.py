import os
import uuid
import zipfile
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Tuple
import shutil
from pydantic import BaseModel
import uvicorn
from app.video_processing import VideoProcessor
from app.image_editing import ImageEditor

app = FastAPI(title="Thumbnail Generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Ensure directories exist
os.makedirs('videos', exist_ok=True)
os.makedirs('thumbnails', exist_ok=True)
os.makedirs('zipped_thumbnails', exist_ok=True)

video_processor = VideoProcessor()
image_editor = ImageEditor()


class EditThumbnailRequest(BaseModel):
    thumbnail_path: str
    text: Optional[str] = None
    font_size: Optional[int] = 40
    text_color: Optional[Tuple[int, int, int]] = (255, 255, 255)
    brightness_adjustment: Optional[float] = 0
    upscale: Optional[bool] = True


@app.post("/edit_thumbnail/")
async def edit_thumbnail(request: EditThumbnailRequest):
    """
    Edit a specific thumbnail
    """
    try:
        edited_image = image_editor.edit_thumbnail(
            thumbnail_path=request.thumbnail_path,
            text=request.text,
            font_size=request.font_size,
            text_color=request.text_color,
            brightness_adjustment=request.brightness_adjustment,
            upscale=request.upscale
        )

        # Save edited image
        edited_path = os.path.join('thumbnails', f'edited_{uuid.uuid4()}.jpg')
        edited_image.save(edited_path)

        return {"edited_thumbnail_path": edited_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_thumbnails/")
async def generate_thumbnails(file: UploadFile = File(...)):
    """
    Generate 5 random thumbnails from an uploaded video
    """
    video_path = os.path.join('videos', f"{uuid.uuid4()}_{file.filename}")
    with open(video_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Generate thumbnails
        thumbnail_tuples = video_processor.extract_thumbnails(video_path)

        # Save generated thumbnails
        thumbnail_paths = []
        for i, (thumbnail, _) in enumerate(thumbnail_tuples):  # Unpack the tuple here
            thumbnail_path = os.path.join('thumbnails', f'thumbnail_{i + 1}.jpg')
            thumbnail.save(thumbnail_path)  # Save the PIL Image directly
            thumbnail_paths.append(thumbnail_path)

        # Create a zip file of thumbnails
        zip_filename = f"thumbnails_{uuid.uuid4()}.zip"
        zip_path = os.path.join('zipped_thumbnails', zip_filename)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for thumbnail_path in thumbnail_paths:
                zipf.write(thumbnail_path, os.path.basename(thumbnail_path))

        return JSONResponse(content={
            "thumbnails": [
                {
                    "id": i + 1,
                    "path": path
                }
                for i, path in enumerate(thumbnail_paths)
            ],
            "zip_filename": os.path.basename(zip_path)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the original video file
        if os.path.exists(video_path):
            os.remove(video_path)


@app.get("/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    """
    Serve a specific thumbnail
    """
    thumbnail_path = os.path.join('thumbnails', filename)
    if not os.path.exists(thumbnail_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumbnail_path)


@app.get("/download_zip/{filename}")
async def download_zip(filename: str, background_tasks: BackgroundTasks):
    """
    Serve the zip file of thumbnails and delete it after download
    """
    zip_path = os.path.join('zipped_thumbnails', filename)
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Zip file not found")

    background_tasks.add_task(os.remove, zip_path)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=filename
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
