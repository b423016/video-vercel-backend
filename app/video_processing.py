import cv2
import numpy as np
from typing import List
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from multiprocessing import Pool


class VideoProcessor:
    def __init__(self):
        # Initialize lightweight feature extraction model (MobileNetV2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_model = mobilenet_v2(pretrained=True).features.eval().to(self.device)

        # Pre-trained emotion model (ResNet18 fine-tuned for emotion recognition)
        self.emotion_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.emotion_model.fc = torch.nn.Linear(self.emotion_model.fc.in_features, 7)  # 7 emotion classes
        self.emotion_model.eval().to(self.device)

        # Transformation for model input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract_features_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        tensors = torch.stack([
            self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames
        ]).to(self.device)

        with torch.no_grad():
            features = self.feature_model(tensors).mean([2, 3]).cpu().numpy()  # Global average pooling

        return features

    def detect_frame_emotions(self, frames: List[np.ndarray]) -> List[str]:
        """
        Detect emotions in a batch of video frames.

        Args:
            frames (List[np.ndarray]): List of input frames (BGR format)

        Returns:
            List[str]: List of predicted emotions for each frame
        """
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        tensors = torch.stack([
            self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames
        ]).to(self.device)

        with torch.no_grad():
            outputs = self.emotion_model(tensors)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        return [emotion_labels[pred] for pred in predictions]

    def _calculate_frame_importance(self, video_path: str, frame_interval: int) -> List[float]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_scores = []
        frames = []

        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            if len(frames) == 32 or i + frame_interval >= total_frames:  # Batch size: 32
                batch_features = self._extract_features_batch(frames)
                for features in batch_features:
                    feature_magnitude = np.linalg.norm(features)
                    frame_scores.append(feature_magnitude)
                frames = []

        cap.release()
        return frame_scores

    def extract_thumbnails(self, video_path: str, num_thumbnails: int = 5, frame_interval: int = 30) -> List:
        """
        Extract key frames and detect emotions for selected thumbnails.

        Args:
            video_path (str): Path to the video file
            num_thumbnails (int): Number of thumbnails to extract
            frame_interval (int): Interval for frame extraction

        Returns:
            List[Tuple[Image.Image, str]]: List of extracted thumbnails with emotions
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame importance
        frame_importance = self._calculate_frame_importance(video_path, frame_interval)

        # Select top N most important frames
        important_frame_indices = sorted(
            range(len(frame_importance)),
            key=lambda i: frame_importance[i],
            reverse=True
        )[:num_thumbnails]

        selected_frames = []
        for frame_index in important_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index * frame_interval)
            ret, frame = cap.read()
            if ret:
                selected_frames.append(frame)

        cap.release()

        # Detect emotions for selected frames
        emotions = self.detect_frame_emotions(selected_frames)
        thumbnails = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in selected_frames
        ]

        return list(zip(thumbnails, emotions))

    def process_in_parallel(self, video_path: str, num_thumbnails: int = 5, frame_interval: int = 30) -> List:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, frame_interval))

        def process_frame(index):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            return frame

        with Pool(processes=4) as pool:
            frames = pool.map(process_frame, frame_indices)

        frames = [frame for frame in frames if frame is not None]
        return self.extract_thumbnails(video_path, num_thumbnails, frame_interval)


# Example Usage
# if __name__ == "__main__":
#     video_processor = VideoProcessor()
#     results = video_processor.extract_thumbnails_with_emotions("video.mp4", num_thumbnails=5, frame_interval=30)
#     for idx, (thumbnail, emotion) in enumerate(results):
#         thumbnail.save(f"thumbnail_{idx + 1}.jpg")
#         print(f"Thumbnail {idx + 1}: Detected Emotion - {emotion}")
