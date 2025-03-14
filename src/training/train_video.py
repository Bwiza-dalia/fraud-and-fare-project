import os
import tensorflow as tf
import logging
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from datetime import datetime

# Import project modules
from src.models.video_model import create_vivit_model, create_3d_resnet_model, create_slowfast_model
from src.preprocessing.video_preprocessing import create_video_dataset
import config

# Configure logging
logger = logging.getLogger(__name__)

def train_video_model(model_type='vivit', dataset_path=None, model_save_path=None, epochs=None, batch_size=None):
    """
    Train a video model for fraud detection.
    
    Args:
        model_type: Type of model to train ('vivit', '3d_resnet', 'slowfast')
        dataset_path: Path to the dataset directory
        model_save_path: Path to save the trained model
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Use default paths if not provided
    if dataset_path is None:
        dataset_path = config.VIDEO_DATA_PATH
        
    if model_save_path is None:
        model_save_path = os.path.join(config.MODEL_ROOT, f"video_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Use default training parameters if not provided
    if epochs is None:
        epochs = config.VIDEO_CONFIG['epochs']
        
    if batch_size is None:
        batch_size = config.VIDEO_CONFIG['batch_size']
    
    logger.info(f"Training {model_type} video model with {epochs} epochs and batch size {batch_size}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_video_dataset(
        dataset_path,
        target_frames=config.VIDEO_CONFIG['target_frames'],
        target_size=config.VIDEO_CONFIG['target_size'],
        batch_size=batch_size
    )
    
    # Create model
    input_shape = (
        config.VIDEO_CONFIG['target_frames'],
        config.VIDEO_CONFIG['target_size'][0],
        config.VIDEO_CONFIG['target_size'][1],
        3  # RGB channels
    )
    
    if model_type == 'vivit':
        model = create_vivit_model(input_shape=input_shape)
    elif model_type == '3d_resnet':
        model = create_3d_resnet_model(input_shape=input_shape)
    elif model_type == 'slowfast':
        model = create_slowfast_model(input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print model summary
    model.summary()
    
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
            log_dir=os.path.join(config.TENSORBOARD_LOG_DIR, f"video_{model_type}"),
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Create class weights to handle imbalanced data
    class_weights = config.VIDEO_CONFIG.get('class_weights', None)
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Evaluate model
    logger.info("Evaluating model on test dataset...")
    test_results = model.evaluate(test_dataset)
    metrics = dict(zip(model.metrics_names, test_results))
    logger.info(f"Test results: {metrics}")
    
    # Save final model
    os.makedirs(model_save_path, exist_ok=True)
    model.save(os.path.join(model_save_path, 'final_model'))
    logger.info(f"Model saved to {model_save_path}")
    
    return model, history

def evaluate_video_model(model_path=None, dataset_path=None, batch_size=None):
    """
    Evaluate a trained video model.
    
    Args:
        model_path: Path to the trained model
        dataset_path: Path to the dataset directory
        batch_size: Batch size for evaluation
        
    Returns:
        metrics: Evaluation metrics
    """
    # Use default paths if not provided
    if model_path is None:
        # Find the most recent model
        model_dir = os.path.join(config.MODEL_ROOT, 'video_model')
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} does not exist")
            return None
        model_path = os.path.join(model_dir, 'final_model')
    
    if dataset_path is None:
        dataset_path = config.VIDEO_DATA_PATH
    
    if batch_size is None:
        batch_size = config.VIDEO_CONFIG['batch_size']
    
    logger.info(f"Evaluating video model from {model_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Create test dataset
    _, _, test_dataset = create_video_dataset(
        dataset_path,
        target_frames=config.VIDEO_CONFIG['target_frames'],
        target_size=config.VIDEO_CONFIG['target_size'],
        batch_size=batch_size
    )
    
    # Evaluate model
    test_results = model.evaluate(test_dataset)
    metrics = dict(zip(model.metrics_names, test_results))
    
    logger.info(f"Test results: {metrics}")
    
    return metrics

def predict_video(model_path, video_path, target_frames=None, target_size=None):
    """
    Make predictions on a single video.
    
    Args:
        model_path: Path to the trained model
        video_path: Path to the video file
        target_frames: Number of frames to extract from the video
        target_size: Target frame size (height, width)
        
    Returns:
        predictions: Model predictions
    """
    # Use default parameters if not provided
    if target_frames is None:
        target_frames = config.VIDEO_CONFIG['target_frames']
    
    if target_size is None:
        target_size = config.VIDEO_CONFIG['target_size']
    
    logger.info(f"Making predictions on video: {video_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Import here to avoid circular imports
    from src.preprocessing.video_preprocessing import extract_frames, preprocess_video
    
    # Extract frames from video
    frames = extract_frames(video_path, target_frames=target_frames, target_size=target_size)
    
    if frames is None:
        logger.error(f"Error extracting frames from {video_path}")
        return None
    
    # Preprocess frames
    frames = preprocess_video(frames, 0, target_frames, target_size, False)[0]
    
    # Add batch dimension
    frames = np.expand_dims(frames, axis=0)
    
    # Make predictions
    predictions = model.predict(frames)
    
    logger.info(f"Predictions: {predictions}")
    
    return predictions

def visualize_model_predictions(model_path, video_path, output_path=None, target_frames=None, target_size=None):
    """
    Visualize model predictions on a video.
    
    Args:
        model_path: Path to the trained model
        video_path: Path to the video file
        output_path: Path to save the output video
        target_frames: Number of frames to extract from the video
        target_size: Target frame size (height, width)
        
    Returns:
        None
    """
    # Use default parameters if not provided
    if target_frames is None:
        target_frames = config.VIDEO_CONFIG['target_frames']
    
    if target_size is None:
        target_size = config.VIDEO_CONFIG['target_size']
    
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(video_path),
            f"predicted_{os.path.basename(video_path)}"
        )
    
    logger.info(f"Visualizing predictions on video: {video_path}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
    # Import here to avoid circular imports
    import cv2
    from src.preprocessing.video_preprocessing import extract_frames, preprocess_video
    
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
    
    # Process video in chunks of target_frames
    for i in range(0, frame_count, target_frames):
        # Extract frames
        frames = []
        for j in range(target_frames):
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
            resized_frame = cv2.resize(frame, target_size)
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(rgb_frame)
        
        # Ensure we have exactly target_frames frames
        if len(processed_frames) < target_frames:
            # Duplicate last frame if needed
            processed_frames.extend([processed_frames[-1]] * (target_frames - len(processed_frames)))
        
        # Convert to numpy array
        processed_frames = np.array(processed_frames)
        
        # Normalize pixel values
        processed_frames = processed_frames.astype(np.float32) / 255.0
        
        # Add batch dimension
        processed_frames = np.expand_dims(processed_frames, axis=0)
        
        # Make predictions
        predictions = model.predict(processed_frames)[0]
        
        # Get prediction class and confidence
        pred_class = np.argmax(predictions)
        confidence = predictions[pred_class]
        
        # Add prediction to frames
        for j, frame in enumerate(frames):
            # Add prediction text
            text = f"Pred: {'Fraud' if pred_class == 1 else 'Legit'} ({confidence:.2f})"
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
    
    # Train a model
    train_video_model(model_type='vivit')