import os
import tensorflow as tf
import logging
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime

# Import project modules
from src.models.video_model import create_vivit_model, load_pretrained_video_model
from src.models.audio_model import create_ast_model, load_pretrained_audio_model
from src.models.fusion_model import (
    create_late_fusion_model, create_early_fusion_model, 
    create_multistream_transformer_fusion, create_weighted_ensemble
)
from src.preprocessing.video_preprocessing import create_video_dataset
from src.preprocessing.audio_preprocessing import create_audio_dataset
import config

# Configure logging
logger = logging.getLogger(__name__)

def create_fusion_datasets(video_dataset_path=None, audio_dataset_path=None, batch_size=None):
    """
    Create datasets for fusion model training.
    
    Args:
        video_dataset_path: Path to video dataset
        audio_dataset_path: Path to audio dataset
        batch_size: Batch size for training
        
    Returns:
        train_video, val_video, test_video, train_audio, val_audio, test_audio: Datasets
    """
    # Use default paths if not provided
    if video_dataset_path is None:
        video_dataset_path = config.VIDEO_DATA_PATH
    
    if audio_dataset_path is None:
        audio_dataset_path = config.AUDIO_DATA_PATH
    
    if batch_size is None:
        batch_size = config.FUSION_CONFIG['batch_size']
    
    logger.info("Creating fusion datasets...")
    
    # Create video datasets
    train_video, val_video, test_video = create_video_dataset(
        video_dataset_path,
        target_frames=config.VIDEO_CONFIG['target_frames'],
        target_size=config.VIDEO_CONFIG['target_size'],
        batch_size=batch_size
    )
    
    # Create audio datasets
    train_audio, val_audio, test_audio = create_audio_dataset(
        audio_dataset_path,
        sr=config.AUDIO_CONFIG['sr'],
        duration=config.AUDIO_CONFIG['duration'],
        n_mels=config.AUDIO_CONFIG['n_mels'],
        n_fft=config.AUDIO_CONFIG['n_fft'],
        hop_length=config.AUDIO_CONFIG['hop_length'],
        batch_size=batch_size
    )
    
    return train_video, val_video, test_video, train_audio, val_audio, test_audio

def create_synchronized_dataset(video_dataset, audio_dataset):
    """
    Create a synchronized dataset for fusion model training.
    This assumes that video_dataset and audio_dataset have the same number of samples
    and the same order of samples.
    
    Args:
        video_dataset: TensorFlow dataset for video data
        audio_dataset: TensorFlow dataset for audio data
        
    Returns:
        synchronized_dataset: Synchronized dataset
    """
    # Convert datasets to numpy arrays
    video_data = []
    audio_data = []
    labels = []
    
    # Extract data from video dataset
    for video, label in video_dataset:
        video_data.append(video.numpy())
        labels.append(label.numpy())
    
    # Extract data from audio dataset
    for audio, _ in audio_dataset:
        audio_data.append(audio.numpy())
    
    # Convert to numpy arrays
    video_data = np.array(video_data)
    audio_data = np.array(audio_data)
    labels = np.array(labels)
    
    # Create synchronized dataset
    synchronized_dataset = tf.data.Dataset.from_tensor_slices(
        ((video_data, audio_data), labels)
    )
    
    return synchronized_dataset

class MultimodalDataGenerator:
    """
    Data generator for synchronized video and audio data.
    """
    def __init__(self, video_dataset, audio_dataset, batch_size=16):
        self.video_dataset = video_dataset
        self.audio_dataset = audio_dataset
        self.batch_size = batch_size
        
        # Get video and audio iterators
        self.video_iterator = iter(self.video_dataset)
        self.audio_iterator = iter(self.audio_dataset)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Get next batch of video data
        try:
            video_batch, labels = next(self.video_iterator)
        except StopIteration:
            # Reset iterators
            self.video_iterator = iter(self.video_dataset)
            self.audio_iterator = iter(self.audio_dataset)
            video_batch, labels = next(self.video_iterator)
        
        # Get next batch of audio data
        try:
            audio_batch, _ = next(self.audio_iterator)
        except StopIteration:
            # Reset audio iterator
            self.audio_iterator = iter(self.audio_dataset)
            audio_batch, _ = next(self.audio_iterator)
        
        return [video_batch, audio_batch], labels

def train_fusion_model(fusion_type='late', video_model_path=None, audio_model_path=None, 
                     video_dataset_path=None, audio_dataset_path=None, 
                     model_save_path=None, epochs=None, batch_size=None):
    """
    Train a fusion model for fraud detection.
    
    Args:
        fusion_type: Type of fusion model ('late', 'early', 'transformer')
        video_model_path: Path to pretrained video model
        audio_model_path: Path to pretrained audio model
        video_dataset_path: Path to video dataset
        audio_dataset_path: Path to audio dataset
        model_save_path: Path to save the trained model
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Use default paths if not provided
    if video_model_path is None:
        video_model_path = os.path.join(config.MODEL_ROOT, 'video_vivit/final_model')
    
    if audio_model_path is None:
        audio_model_path = os.path.join(config.MODEL_ROOT, 'audio_ast/final_model')
    
    if model_save_path is None:
        model_save_path = os.path.join(config.MODEL_ROOT, f"fusion_{fusion_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Use default training parameters if not provided
    if epochs is None:
        epochs = config.FUSION_CONFIG['epochs']
        
    if batch_size is None:
        batch_size = config.FUSION_CONFIG['batch_size']
    
    logger.info(f"Training {fusion_type} fusion model with {epochs} epochs and batch size {batch_size}")
    
    # Create datasets
    train_video, val_video, test_video, train_audio, val_audio, test_audio = create_fusion_datasets(
        video_dataset_path=video_dataset_path,
        audio_dataset_path=audio_dataset_path,
        batch_size=batch_size
    )
    
    # Create model
    if fusion_type == 'late':
        # Load pretrained models
        logger.info("Loading pretrained video model...")
        video_model = load_pretrained_video_model(video_model_path)
        if video_model is None:
            logger.error("Failed to load video model")
            return None, None
        
        logger.info("Loading pretrained audio model...")
        audio_model = load_pretrained_audio_model(audio_model_path)
        if audio_model is None:
            logger.error("Failed to load audio model")
            return None, None
        
        # Create late fusion model
        model = create_late_fusion_model(video_model, audio_model)
    
    elif fusion_type == 'early':
        # Create early fusion model
        video_shape = (
            config.VIDEO_CONFIG['target_frames'],
            config.VIDEO_CONFIG['target_size'][0],
            config.VIDEO_CONFIG['target_size'][1],
            3  # RGB channels
        )
        
        audio_shape = (
            config.AUDIO_CONFIG['n_mels'],
            int(np.ceil(config.AUDIO_CONFIG['sr'] * config.AUDIO_CONFIG['duration'] / config.AUDIO_CONFIG['hop_length'])),
            1  # Channels
        )
        
        model = create_early_fusion_model(video_shape=video_shape, audio_shape=audio_shape)
    
    elif fusion_type == 'transformer':
        # Create transformer fusion model
        video_shape = (
            config.VIDEO_CONFIG['target_frames'],
            config.VIDEO_CONFIG['target_size'][0],
            config.VIDEO_CONFIG['target_size'][1],
            3  # RGB channels
        )
        
        audio_shape = (
            config.AUDIO_CONFIG['n_mels'],
            int(np.ceil(config.AUDIO_CONFIG['sr'] * config.AUDIO_CONFIG['duration'] / config.AUDIO_CONFIG['hop_length'])),
            1  # Channels
        )
        
        model = create_multistream_transformer_fusion(video_shape=video_shape, audio_shape=audio_shape)
    
    elif fusion_type == 'weighted_ensemble':
        # Load pretrained models
        logger.info("Loading pretrained video model...")
        video_model = load_pretrained_video_model(video_model_path)
        if video_model is None:
            logger.error("Failed to load video model")
            return None, None
        
        logger.info("Loading pretrained audio model...")
        audio_model = load_pretrained_audio_model(audio_model_path)
        if audio_model is None:
            logger.error("Failed to load audio model")
            return None, None
        
        # Create weighted ensemble
        model = create_weighted_ensemble(video_model, audio_model, video_weight=0.6, audio_weight=0.4)
        
        # With weighted ensemble, we just return the ensemble function
        logger.info("Weighted ensemble created")
        return model, None
    
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    # Print model summary
    model.summary()
    
    # Create multimodal data generator
    train_generator = MultimodalDataGenerator(train_video, train_audio, batch_size=batch_size)
    val_generator = MultimodalDataGenerator(val_video, val_audio, batch_size=batch_size)
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_save_path, 'checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.TRAIN_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.TRAIN_CONFIG['reduce_lr_factor'],
            patience=config.TRAIN_CONFIG['reduce_lr_patience'],
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(config.TENSORBOARD_LOG_DIR, f"fusion_{fusion_type}"),
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=len(train_video) // batch_size,
        validation_steps=len(val_video) // batch_size
    )
    
    # Evaluate model
    logger.info("Evaluating model on test dataset...")
    test_generator = MultimodalDataGenerator(test_video, test_audio, batch_size=batch_size)
    test_results = model.evaluate(
        test_generator,
        steps=len(test_video) // batch_size
    )
    
    metrics = dict(zip(model.metrics_names, test_results))
    logger.info(f"Test results: {metrics}")
    
    # Save final model
    os.makedirs(model_save_path, exist_ok=True)
    model.save(os.path.join(model_save_path, 'final_model'))
    logger.info(f"Model saved to {model_save_path}")
    
    return model, history

def evaluate_fusion_model(model_path=None, video_dataset_path=None, audio_dataset_path=None, batch_size=None):
    """
    Evaluate a trained fusion model.
    
    Args:
        model_path: Path to the trained model
        video_dataset_path: Path to video dataset
        audio_dataset_path: Path to audio dataset
        batch_size: Batch size for evaluation
        
    Returns:
        metrics: Evaluation metrics
    """
    # Use default paths if not provided
    if model_path is None:
        # Find the most recent model
        model_dir = os.path.join(config.MODEL_ROOT, 'fusion_model')
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} does not exist")
            return None
        model_path = os.path.join(model_dir, 'final_model')
    
    if batch_size is None:
        batch_size = config.FUSION_CONFIG['batch_size']
    
    logger.info(f"Evaluating fusion model from {model_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Create datasets
    _, _, test_video, _, _, test_audio = create_fusion_datasets(
        video_dataset_path=video_dataset_path,
        audio_dataset_path=audio_dataset_path,
        batch_size=batch_size
    )
    
    # Create test generator
    test_generator = MultimodalDataGenerator(test_video, test_audio, batch_size=batch_size)
    
    # Evaluate model
    test_results = model.evaluate(
        test_generator,
        steps=len(test_video) // batch_size
    )
    
    metrics = dict(zip(model.metrics_names, test_results))
    logger.info(f"Test results: {metrics}")
    
    return metrics

def predict_fusion(model_path, video_path, audio_path):
    """
    Make predictions using a fusion model.
    
    Args:
        model_path: Path to the trained model
        video_path: Path to the video file
        audio_path: Path to the audio file
        
    Returns:
        predictions: Model predictions
    """
    logger.info(f"Making predictions on video: {video_path} and audio: {audio_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Import here to avoid circular imports
    from src.preprocessing.video_preprocessing import extract_frames, preprocess_video
    from src.preprocessing.audio_preprocessing import extract_audio_features
    
    # Extract frames from video
    video_frames = extract_frames(
        video_path, 
        target_frames=config.VIDEO_CONFIG['target_frames'], 
        target_size=config.VIDEO_CONFIG['target_size']
    )
    
    if video_frames is None:
        logger.error(f"Error extracting frames from {video_path}")
        return None
    
    # Preprocess frames
    video_frames = preprocess_video(
        video_frames, 0, 
        config.VIDEO_CONFIG['target_frames'], 
        config.VIDEO_CONFIG['target_size'], 
        False
    )[0]
    
    # Extract features from audio
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
    
    # Add channel dimension to audio
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    
    # Add batch dimension
    video_frames = np.expand_dims(video_frames, axis=0)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    
    # Make predictions
    predictions = model.predict([video_frames, mel_spectrogram])
    
    logger.info(f"Predictions: {predictions}")
    
    return predictions

def visualize_fusion_predictions(model_path, video_path, audio_path, output_path=None):
    """
    Visualize model predictions on a video and audio.
    
    Args:
        model_path: Path to the trained model
        video_path: Path to the video file
        audio_path: Path to the audio file
        output_path: Path to save the output video
        
    Returns:
        None
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(video_path),
            f"fusion_predicted_{os.path.basename(video_path)}"
        )
    
    logger.info(f"Visualizing fusion predictions on video: {video_path} and audio: {audio_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Import here to avoid circular imports
    import cv2
    import librosa
    from src.preprocessing.video_preprocessing import extract_frames, preprocess_video
    from src.preprocessing.audio_preprocessing import extract_audio_features
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=config.AUDIO_CONFIG['sr'])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video: {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video in chunks
    chunk_size = config.VIDEO_CONFIG['target_frames']
    audio_duration = config.AUDIO_CONFIG['duration']
    
    for i in range(0, frame_count, chunk_size):
        # Extract frames
        frames = []
        for j in range(chunk_size):
            if i + j < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i + j)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        
        if not frames:
            continue
        
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            # Resize frame
            resized_frame = cv2.resize(frame, config.VIDEO_CONFIG['target_size'])
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(rgb_frame)
        
        # Ensure we have exactly chunk_size frames
        if len(processed_frames) < chunk_size:
            # Duplicate last frame if needed
            processed_frames.extend([processed_frames[-1]] * (chunk_size - len(processed_frames)))
        
        # Convert to numpy array
        processed_frames = np.array(processed_frames)
        
        # Normalize pixel values
        processed_frames = processed_frames.astype(np.float32) / 255.0
        
        # Get audio for this chunk
        audio_start = int((i / fps) * sr)
        audio_end = int(audio_start + audio_duration * sr)
        
        if audio_end > len(y):
            # Pad with zeros if needed
            chunk_audio = np.pad(y[audio_start:], (0, audio_end - len(y)))
        else:
            chunk_audio = y[audio_start:audio_end]
        
        # Extract mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=chunk_audio,
            sr=sr,
            n_mels=config.AUDIO_CONFIG['n_mels'],
            n_fft=config.AUDIO_CONFIG['n_fft'],
            hop_length=config.AUDIO_CONFIG['hop_length']
        )
        
        # Convert to dB scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize to [0, 1] range
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min() + 1e-8)
        
        # Add channel dimension to audio
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
        
        # Add batch dimension
        processed_frames = np.expand_dims(processed_frames, axis=0)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        
        # Make predictions
        predictions = model.predict([processed_frames, mel_spectrogram])[0]
        
        # Get prediction class and confidence
        pred_class = np.argmax(predictions)
        confidence = predictions[pred_class]
        
        # Add prediction to frames
        for j, frame in enumerate(frames):
            # Add prediction text
            text = f"Fusion Pred: {'Fraud' if pred_class == 1 else 'Legit'} ({confidence:.2f})"
            cv2.putText(
                frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if pred_class == 1 else (0, 255, 0), 2
            )
            
            # Write frame to output video
            out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    logger.info(f"Output video saved to {output_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train a fusion model
    train_fusion_model(fusion_type='late')