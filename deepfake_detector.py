import os
import torch
import torchaudio
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import warnings
warnings.filterwarnings("ignore")

class DeepfakeAudioDetector:
    def __init__(self, model_name="MelodyMachine/Deepfake-audio-detection-V2", device=None):
        """
        Initialize the deepfake audio detector.
        
        Args:
            model_name (str): HuggingFace model path
            device (str): Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model on {self.device}...")
        
        # Load model and feature extractor
        self.model = AutoModelForAudioClassification.from_pretrained(model_name).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Model expects 16kHz audio
        self.sample_rate = 16000
        
        print("Model loaded successfully!")
        
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file for the model.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            torch.Tensor: Processed audio features
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Load and resample audio using librosa for better compatibility
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            # Convert to torch tensor
            waveform = torch.tensor(waveform).unsqueeze(0)
        except Exception as e:
            print(f"Error loading audio with librosa: {e}")
            # Fallback to torchaudio
            waveform, sr = torchaudio.load(audio_path)
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
        # Extract features
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(), 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        return inputs.to(self.device)
    
    def predict(self, audio_path):
        """
        Predict if audio is real or fake.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary with prediction results
        """
        # Preprocess audio
        inputs = self.preprocess_audio(audio_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get logits and convert to probabilities
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get prediction (model has 2 classes: real [0] and fake [1])
        prediction_idx = np.argmax(probabilities)
        is_fake = prediction_idx == 1
        
        # Create result dictionary
        result = {
            "is_fake": bool(is_fake),
            "confidence": float(probabilities[prediction_idx]),
            "real_probability": float(probabilities[0]),
            "fake_probability": float(probabilities[1]),
            "classification": "FAKE" if is_fake else "REAL"
        }
        
        return result
    
    def batch_predict(self, audio_paths):
        """
        Run predictions on multiple audio files.
        
        Args:
            audio_paths (list): List of paths to audio files
            
        Returns:
            dict: Dictionary mapping file paths to prediction results
        """
        results = {}
        for path in audio_paths:
            try:
                results[path] = self.predict(path)
            except Exception as e:
                results[path] = {"error": str(e)}
                
        return results

# Example usage
if __name__ == "__main__":
    detector = DeepfakeAudioDetector()
    
    # Test with a sample file
    test_file = "test_samples/sample.wav"
    if os.path.exists(test_file):
        result = detector.predict(test_file)
        print(f"Results for {test_file}:")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Real probability: {result['real_probability']:.2%}")
        print(f"Fake probability: {result['fake_probability']:.2%}")
    else:
        print(f"Test file {test_file} not found. Please create a test_samples directory with sample audio files.")