import os

project_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory

# Define directory structure (ensure exact paths)
directory_structure = {
    'app': {
        'main.py': "",
        'video_processing.py': "",
        'emotion_model.py': "",
        'image_editing.py': "",
        'models.py': "",
        'utils': {
            'scene_detection.py': "",
            'video_utils.py': "",
        }
    },
    'assets': {
        'emojis': "",
        'fonts': "",
    },
    'videos': "",
    'thumbnails': "",
    'requirements.txt': "",
    'docs': {
        'README.md': "",
    }
}

# Create directories with exact paths
for directory, sub_structure in directory_structure.items():
    path = os.path.join(project_dir, directory)
    os.makedirs(path, exist_ok=True)  # Create directories if they don't exist

    if sub_structure:
        for filename, _ in sub_structure.items():
            filepath = os.path.join(path, filename)
            with open(filepath, 'w') as f:
                f.write("")  # Create empty files

print("Project structure created successfully!")