import os
import argparse
from deepfake_detector import DeepfakeAudioDetector

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test deepfake audio detection')
    parser.add_argument('audio_path', type=str, help='Path to the audio file to analyze')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                        help='Device to run inference on (default: auto-detect)')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.audio_path):
        print(f"Error: File '{args.audio_path}' does not exist.")
        return
    
    # Initialize detector
    print(f"Loading deepfake audio detector...")
    detector = DeepfakeAudioDetector(device=args.device)
    
    # Run prediction
    print(f"Analyzing audio file: {args.audio_path}")
    result = detector.predict(args.audio_path)
    
    # Print results
    print("\n" + "="*50)
    print("DEEPFAKE AUDIO DETECTION RESULTS")
    print("="*50)
    print(f"Audio file: {args.audio_path}")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Real probability: {result['real_probability']:.2%}")
    print(f"Fake probability: {result['fake_probability']:.2%}")
    print("="*50)
    
    return result

if __name__ == "__main__":
    main()