import os
import numpy as np
import tensorflow as tf
import cv2
import librosa
import logging
import time
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

# Import project modules
from src.preprocessing.video_preprocessing import extract_frames, preprocess_video
from src.preprocessing.audio_preprocessing import extract_audio_features
import config

# Configure logging
logger = logging.getLogger(__name__)

def load_video_model(model_path=None):
    """
    Load a trained video model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        model: Loaded model
    """
    # Use default path if not provided
    if model_path is None:
        model_path = os.path.join(config.MODEL_ROOT, 'video_model', 'final_model')
    
    logger.info(f"Loading video model from {model_path}")
    
    try:
        # Define custom objects
        custom_objects = {
            'F1Score': tfa.metrics.F1Score,
            'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy
        }
        
        # Load model
        model = load_model(model_path, custom_objects=custom_objects)
        logger.info("Video model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading video model: {e}")
        return None

def load_audio_model(model_path=None):
    """
    Load a trained audio model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        model: Loaded model
    """
    # Use default path if not provided
    if model_path is None:
        model_path = os.path.join(config.MODEL_ROOT, 'audio_model', 'final_model')
    
    logger.info(f"Loading audio model from {model_path}")
    
    try:
        # Define custom objects
        custom_objects = {
            'F1Score': tfa.metrics.F1Score,
            'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy
        }
        
        # Load model
        model = load_model(model_path, custom_objects=custom_objects)
        logger.info("Audio model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading audio model: {e}")
        return None

def load_fusion_model(model_path=None):
    """
    Load a trained fusion model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        model: Loaded model
    """
    # Use default path if not provided
    if model_path is None:
        model_path = os.path.join(config.MODEL_ROOT, 'fusion_model', 'final_model')
    
    logger.info(f"Loading fusion model from {model_path}")
    
    try:
        # Define custom objects
        custom_objects = {
            'F1Score': tfa.metrics.F1Score,
            'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy
        }
        
        # Load model
        model = load_model(model_path, custom_objects=custom_objects)
        logger.info("Fusion model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading fusion model: {e}")
        return None

def preprocess_video_for_inference(video_path):
    """
    Preprocess a video for model inference.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        frames: Preprocessed video frames
    """
    logger.info(f"Preprocessing video: {video_path}")
    
    try:
        # Extract frames
        frames = extract_frames(
            video_path,
            target_frames=config.VIDEO_CONFIG['target_frames'],
            target_size=config.VIDEO_CONFIG['target_size']
        )
        
        if frames is None:
            logger.error(f"Error extracting frames from {video_path}")
            return None
        
        # Convert to numpy array
        frames = np.array(frames)
        
        # Normalize pixel values
        frames = frames.astype(np.float32) / 255.0
        
        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)
        
        return frames
    
    except Exception as e:
        logger.error(f"Error preprocessing video: {e}")
        return None

def preprocess_audio_for_inference(audio_path):
    """
    Preprocess an audio file for model inference.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        mel_spectrogram: Preprocessed mel spectrogram
    """
    logger.info(f"Preprocessing audio: {audio_path}")
    
    try:
        # Extract mel spectrogram
        mel_spectrogram = extract_audio_features(
            audio_path,
            sr=config.AUDIO_CONFIG['sr'],
            duration=config.AUDIO_CONFIG['duration'],
            n_mels=config.AUDIO_CONFIG['n_mels'],
            n_fft=config.AUDIO_CONFIG['n_fft'],
            hop_length=config.AUDIO_CONFIG['hop_length']
        )
        
        if mel_spectrogram is None:
            logger.error(f"Error extracting features from {audio_path}")
            return None
        
        # Add channel dimension
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
        
        # Add batch dimension
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        
        return mel_spectrogram
    
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return None

def predict_video(model, video_path, threshold=0.5):
    """
    Make prediction on a video using a trained model.
    
    Args:
        model: Trained model
        video_path: Path to video file
        threshold: Threshold for fraud classification
        
    Returns:
        result: Dictionary with prediction results
    """
    logger.info(f"Predicting using video model: {video_path}")
    
    try:
        # Start timer
        start_time = time.time()
        
        # Preprocess video
        frames = preprocess_video_for_inference(video_path)
        
        if frames is None:
            return {'error': 'Error preprocessing video'}
        
        # Make prediction
        prediction = model.predict(frames)[0]
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        
        # Create result dictionary
        result = {
            'prediction': prediction.tolist(),
            'fraud_probability': float(prediction[1]),
            'fraud_detected': bool(prediction[1] > threshold),
            'prediction_time': prediction_time
        }
        
        logger.info(f"Prediction result: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting video: {e}")
        return {'error': str(e)}

def predict_audio(model, audio_path, threshold=0.5):
    """
    Make prediction on an audio file using a trained model.
    
    Args:
        model: Trained model
        audio_path: Path to audio file
        threshold: Threshold for fraud classification
        
    Returns:
        result: Dictionary with prediction results
    """
    logger.info(f"Predicting using audio model: {audio_path}")
    
    try:
        # Start timer
        start_time = time.time()
        
        # Preprocess audio
        mel_spectrogram = preprocess_audio_for_inference(audio_path)
        
        if mel_spectrogram is None:
            return {'error': 'Error preprocessing audio'}
        
        # Make prediction
        prediction = model.predict(mel_spectrogram)[0]
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        
        # Create result dictionary
        result = {
            'prediction': prediction.tolist(),
            'fraud_probability': float(prediction[1]),
            'fraud_detected': bool(prediction[1] > threshold),
            'prediction_time': prediction_time
        }
        
        logger.info(f"Prediction result: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting audio: {e}")
        return {'error': str(e)}

def predict_fusion(model, video_path, audio_path, threshold=0.5):
    """
    Make prediction using a fusion model.
    
    Args:
        model: Trained fusion model
        video_path: Path to video file
        audio_path: Path to audio file
        threshold: Threshold for fraud classification
        
    Returns:
        result: Dictionary with prediction results
    """
    logger.info(f"Predicting using fusion model: {video_path}, {audio_path}")
    
    try:
        # Start timer
        start_time = time.time()
        
        # Preprocess video
        video_frames = preprocess_video_for_inference(video_path)
        
        if video_frames is None:
            return {'error': 'Error preprocessing video'}
        
        # Preprocess audio
        mel_spectrogram = preprocess_audio_for_inference(audio_path)
        
        if mel_spectrogram is None:
            return {'error': 'Error preprocessing audio'}
        
        # Make prediction
        prediction = model.predict([video_frames, mel_spectrogram])[0]
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        
        # Create result dictionary
        result = {
            'prediction': prediction.tolist(),
            'fraud_probability': float(prediction[1]),
            'fraud_detected': bool(prediction[1] > threshold),
            'prediction_time': prediction_time
        }
        
        logger.info(f"Prediction result: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting fusion: {e}")
        return {'error': str(e)}

def ensemble_predict(video_model, audio_model, video_path, audio_path, video_weight=0.6, audio_weight=0.4, threshold=0.5):
    """
    Make prediction using ensemble of video and audio models.
    
    Args:
        video_model: Trained video model
        audio_model: Trained audio model
        video_path: Path to video file
        audio_path: Path to audio file
        video_weight: Weight for video model prediction
        audio_weight: Weight for audio model prediction
        threshold: Threshold for fraud classification
        
    Returns:
        result: Dictionary with prediction results
    """
    logger.info(f"Predicting using ensemble: {video_path}, {audio_path}")
    
    try:
        # Start timer
        start_time = time.time()
        
        # Get video prediction
        video_result = predict_video(video_model, video_path, threshold=0)
        
        if 'error' in video_result:
            return video_result
        
        # Get audio prediction
        audio_result = predict_audio(audio_model, audio_path, threshold=0)
        
        if 'error' in audio_result:
            return audio_result
        
        # Weighted average of predictions
        video_pred = np.array(video_result['prediction'])
        audio_pred = np.array(audio_result['prediction'])
        
        ensemble_pred = video_pred * video_weight + audio_pred * audio_weight
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        
        # Create result dictionary
        result = {
            'prediction': ensemble_pred.tolist(),
            'fraud_probability': float(ensemble_pred[1]),
            'fraud_detected': bool(ensemble_pred[1] > threshold),
            'video_fraud_probability': float(video_pred[1]),
            'audio_fraud_probability': float(audio_pred[1]),
            'prediction_time': prediction_time
        }
        
        logger.info(f"Ensemble prediction result: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting ensemble: {e}")
        return {'error': str(e)}

def run_inference(video_path=None, audio_path=None, model_type='fusion', threshold=0.7):
    """
    Run inference using the specified model type.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        model_type: Type of model to use ('video', 'audio', 'fusion', 'ensemble')
        threshold: Threshold for fraud classification
        
    Returns:
        result: Dictionary with prediction results
    """
    logger.info(f"Running inference with model type: {model_type}")
    
    try:
        # Load appropriate model(s) based on model_type
        if model_type == 'video':
            if video_path is None:
                return {'error': 'Video path not provided'}
            
            # Load video model
            model = load_video_model()
            
            if model is None:
                return {'error': 'Failed to load video model'}
            
            # Run inference
            return predict_video(model, video_path, threshold)
        
        elif model_type == 'audio':
            if audio_path is None:
                return {'error': 'Audio path not provided'}
            
            # Load audio model
            model = load_audio_model()
            
            if model is None:
                return {'error': 'Failed to load audio model'}
            
            # Run inference
            return predict_audio(model, audio_path, threshold)
        
        elif model_type == 'fusion':
            if video_path is None:
                return {'error': 'Video path not provided'}
            
            if audio_path is None:
                return {'error': 'Audio path not provided'}
            
            # Load fusion model
            model = load_fusion_model()
            
            if model is None:
                return {'error': 'Failed to load fusion model'}
            
            # Run inference
            return predict_fusion(model, video_path, audio_path, threshold)
        
        elif model_type == 'ensemble':
            if video_path is None:
                return {'error': 'Video path not provided'}
            
            if audio_path is None:
                return {'error': 'Audio path not provided'}
            
            # Load video and audio models
            video_model = load_video_model()
            audio_model = load_audio_model()
            
            if video_model is None:
                return {'error': 'Failed to load video model'}
            
            if audio_model is None:
                return {'error': 'Failed to load audio model'}
            
            # Run inference
            return ensemble_predict(video_model, audio_model, video_path, audio_path, threshold=threshold)
        
        else:
            return {'error': f'Unknown model type: {model_type}'}
    
    except Exception as e:
        logger.error(f"Error running inference: {e}")
        return {'error': str(e)}

def real_time_detection(video_source=0, model_type='video', threshold=0.7, display=True):
    """
    Run real-time fraud detection on a video stream.
    
    Args:
        video_source: Video source (0 for webcam, or path to video file)
        model_type: Type of model to use ('video', 'audio', 'fusion', 'ensemble')
        threshold: Threshold for fraud classification
        display: Whether to display the video with prediction results
        
    Returns:
        None
    """
    logger.info(f"Starting real-time detection with model type: {model_type}")
    
    try:
        # Load appropriate model(s) based on model_type
        if model_type == 'video':
            # Load video model
            model = load_video_model()
            
            if model is None:
                logger.error("Failed to load video model")
                return
        
        elif model_type in ['fusion', 'ensemble', 'audio']:
            logger.error(f"Real-time detection with {model_type} model not supported")
            return
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return
        
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"Error opening video source: {video_source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Buffer to store frames
        frame_buffer = []
        buffer_size = config.VIDEO_CONFIG['target_frames']
        
        # Main loop
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                logger.info("End of video stream")
                break
            
            # Resize frame
            frame_resized = cv2.resize(frame, config.VIDEO_CONFIG['target_size'])
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Add to buffer
            frame_buffer.append(frame_rgb)
            
            # Keep buffer at target size
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
            
            # Make prediction when buffer is full
            if len(frame_buffer) == buffer_size:
                # Prepare frames for prediction
                frames = np.array(frame_buffer)
                frames = frames.astype(np.float32) / 255.0
                frames = np.expand_dims(frames, axis=0)
                
                # Make prediction
                prediction = model.predict(frames, verbose=0)[0]
                
                # Get fraud probability
                fraud_prob = prediction[1]
                
                # Determine fraud status
                fraud_detected = fraud_prob > threshold
                
                # Add prediction to frame
                text = f"Fraud: {fraud_prob:.2f}"
                color = (0, 0, 255) if fraud_detected else (0, 255, 0)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Display status
                status_text = "FRAUD DETECTED!" if fraud_detected else "Normal"
                cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display frame
            if display:
                cv2.imshow('Fraud Detection', frame)
                
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release resources
        cap.release()
        
        if display:
            cv2.destroyAllWindows()
        
        logger.info("Real-time detection stopped")
    
    except Exception as e:
        logger.error(f"Error in real-time detection: {e}")

def batch_inference(input_dir, output_path=None, model_type='video', threshold=0.7):
    """
    Run batch inference on a directory of videos.
    
    Args:
        input_dir: Directory containing video files
        output_path: Path to save the inference results
        model_type: Type of model to use ('video', 'audio', 'fusion', 'ensemble')
        threshold: Threshold for fraud classification
        
    Returns:
        results: Dictionary with prediction results for each file
    """
    logger.info(f"Running batch inference on {input_dir} with model type: {model_type}")
    
    # Use default output path if not provided
    if output_path is None:
        output_path = os.path.join(input_dir, 'inference_results.csv')
    
    try:
        # Get list of files
        files = []
        
        # Check file extensions based on model type
        if model_type == 'video':
            valid_extensions = ['.mp4', '.avi', '.mov']
            for file in os.listdir(input_dir):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    files.append(os.path.join(input_dir, file))
        
        elif model_type == 'audio':
            valid_extensions = ['.wav', '.mp3', '.ogg']
            for file in os.listdir(input_dir):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    files.append(os.path.join(input_dir, file))
        
        elif model_type in ['fusion', 'ensemble']:
            logger.error(f"Batch inference with {model_type} model not supported")
            return None
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        if not files:
            logger.error(f"No valid files found in {input_dir}")
            return None
        
        # Load appropriate model(s) based on model_type
        if model_type == 'video':
            # Load video model
            model = load_video_model()
            
            if model is None:
                logger.error("Failed to load video model")
                return None
            
            # Process each file
            results = {}
            for file in files:
                logger.info(f"Processing: {file}")
                result = predict_video(model, file, threshold)
                results[file] = result
        
        elif model_type == 'audio':
            # Load audio model
            model = load_audio_model()
            
            if model is None:
                logger.error("Failed to load audio model")
                return None
            
            # Process each file
            results = {}
            for file in files:
                logger.info(f"Processing: {file}")
                result = predict_audio(model, file, threshold)
                results[file] = result
        
        # Save results to CSV
        import pandas as pd
        
        # Create dataframe
        df_data = []
        for file, result in results.items():
            if 'error' in result:
                continue
            
            df_data.append({
                'file': os.path.basename(file),
                'fraud_probability': result['fraud_probability'],
                'fraud_detected': result['fraud_detected'],
                'prediction_time': result['prediction_time']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Batch inference results saved to {output_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in batch inference: {e}")
        return None

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference for fraud detection')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--model_type', type=str, default='video', 
                        choices=['video', 'audio', 'fusion', 'ensemble'],
                        help='Type of model to use')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold for fraud classification')
    parser.add_argument('--batch', action='store_true',
                        help='Run batch inference on a directory')
    parser.add_argument('--realtime', action='store_true',
                        help='Run real-time detection on webcam or video')
    
    args = parser.parse_args()
    
    if args.realtime:
        video_source = 0  # Webcam
        if args.video:
            video_source = args.video
        
        real_time_detection(
            video_source=video_source,
            model_type=args.model_type,
            threshold=args.threshold
        )
    
    elif args.batch:
        if args.video:
            batch_inference(
                input_dir=args.video,
                model_type=args.model_type,
                threshold=args.threshold
            )
        else:
            logger.error("Input directory not provided for batch inference")
    
    else:
        # Run single inference
        result = run_inference(
            video_path=args.video,
            audio_path=args.audio,
            model_type=args.model_type,
            threshold=args.threshold
        )
        
        print(result)