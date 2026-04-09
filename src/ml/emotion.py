from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


@dataclass(frozen=True)
class EmotionResult:
    label: str
    confidence: float


class EmotionDetector:
    """Emotion detector using facial feature detection (mouth, eyes, face geometry)."""
    
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path
        self.labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        
        # Load cascade classifiers
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cascade_dir + 'haarcascade_smile.xml')

    def predict(self, image: Image.Image | np.ndarray) -> EmotionResult:
        """Predict emotion based on facial features (mouth opening, eye openness)."""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image.astype(np.uint8)
            
            # Ensure RGB format (remove alpha if present)
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                return EmotionResult(label="neutral", confidence=0.5)
            
            # Use first detected face
            (x, y, w, h) = faces[0]
            face_region = gray[y:y+h, x:x+w]
            
            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
            eye_count = len(eyes)
            
            # Detect smile/mouth in face region
            smiles = self.mouth_cascade.detectMultiScale(face_region, scaleFactor=1.7, minNeighbors=20, minSize=(20, 20))
            smile_detected = len(smiles) > 0
            
            # Analyze face region brightness for additional clues
            brightness = np.mean(face_region) / 255.0
            
            # Simple emotion classification based on features
            if smile_detected and eye_count >= 2:
                emotion = "happy"
                confidence = 0.80
            elif eye_count >= 2 and brightness > 0.6:
                emotion = "neutral"
                confidence = 0.70
            elif eye_count >= 2 and brightness < 0.4:
                emotion = "sad"
                confidence = 0.68
            elif eye_count == 0:
                # Eyes not detected, likely closed or squinting (sad/tired)
                emotion = "sad"
                confidence = 0.65
            else:
                emotion = "neutral"
                confidence = 0.60
            
            return EmotionResult(label=emotion, confidence=confidence)
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return EmotionResult(label="neutral", confidence=0.5)
