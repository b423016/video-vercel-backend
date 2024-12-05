import cv2
import numpy as np
from typing import List
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, ResNetModel, AutoModel
import torch.nn.functional as F


class VideoProcessor:
    def __init__(self):
        # Initialize scene detection and emotion detection models
        self.scene_detection_model = cv2.createBackgroundSubtractorMOG2()

        # Pretrained emotion detection model
        self.emotion_model_name = "dima806/facial_emotions_image_detection"
        self.emotion_extractor = AutoFeatureExtractor.from_pretrained(self.emotion_model_name)
        self.emotion_model = AutoModelForImageClassification.from_pretrained(self.emotion_model_name)

        # Additional feature extraction models
        self.feature_extractors = [
            # ResNet model for feature extraction
            {
                'model': ResNetModel.from_pretrained('microsoft/resnet-50'),
                'name': 'resnet50',
                'input_size': (224, 224)
            },
            # Another pre-trained vision transformer
            {
                'model': AutoModel.from_pretrained('google/vit-base-patch16-224'),
                'name': 'vision_transformer',
                'input_size': (224, 224)
            }
        ]

        # Transformation for model input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract_features_with_multiple_models(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a frame using multiple pre-trained models

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Combined feature vector
        """
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Aggregate features from multiple models
        combined_features = []

        for extractor in self.feature_extractors:
            # Prepare input for the model
            inputs = self.transform(pil_image).unsqueeze(0)

            # Extract features
            with torch.no_grad():
                if extractor['name'] == 'resnet50':
                    outputs = extractor['model'](inputs)
                    features = outputs.pooler_output.squeeze().numpy()
                elif extractor['name'] == 'vision_transformer':
                    outputs = extractor['model'](inputs)
                    features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

                # Normalize features
                features = features / np.linalg.norm(features)
                combined_features.append(features)

        # Combine features from different models
        combined_features = np.concatenate(combined_features)
        return combined_features

    def _calculate_frame_importance(self, video_path: str) -> List[float]:
        """
        Calculate frame importance scores using multiple AI models

        Args:
            video_path (str): Path to the video file

        Returns:
            List[float]: Importance scores for each frame
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_scores = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Extract features using multiple models
            frame_features = self._extract_features_with_multiple_models(frame)

            # Calculate importance score based on feature magnitude and diversity
            feature_magnitude = np.linalg.norm(frame_features)
            feature_entropy = self._calculate_feature_entropy(frame_features)

            # Combined scoring mechanism
            importance_score = feature_magnitude * feature_entropy
            frame_scores.append(importance_score)

        cap.release()
        return frame_scores

    def _calculate_feature_entropy(self, features: np.ndarray) -> float:
        """
        Calculate entropy of feature vector to measure information content

        Args:
            features (np.ndarray): Input feature vector

        Returns:
            float: Entropy value
        """
        # Normalize features
        normalized_features = features / np.linalg.norm(features)

        # Calculate probability distribution
        probabilities = np.abs(normalized_features)
        probabilities /= probabilities.sum()

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def extract_thumbnails(self, video_path: str, num_thumbnails: int = 5) -> List[Image.Image]:
        """
        Extract key frames from video using advanced AI-based techniques

        Args:
            video_path (str): Path to the video file
            num_thumbnails (int): Number of thumbnails to extract

        Returns:
            List[Image.Image]: List of extracted thumbnails
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # AI-based frame selection
        frame_importance = self._calculate_frame_importance(video_path)

        # Select top N most important frames
        important_frame_indices = sorted(
            range(len(frame_importance)),
            key=lambda i: frame_importance[i],
            reverse=True
        )[:num_thumbnails]

        selected_frames = []
        for frame_index in important_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                selected_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        cap.release()
        return selected_frames

    def detect_frame_emotions(self, image_path: str) -> dict:
        """
        Detect emotions in a frame

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: Detected emotion and confidence
        """
        image = Image.open(image_path)
        inputs = self.emotion_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        emotions = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }

        return {
            "emotion": emotions.get(predicted_class_idx, "Unknown"),
            "confidence": float(torch.softmax(logits, dim=1).max())
        }