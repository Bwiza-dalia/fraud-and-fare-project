import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import tensorflow_addons as tfa
import logging
import numpy as np

# Import video and audio models
from src.models.video_model import create_vivit_model, load_pretrained_video_model
from src.models.audio_model import create_ast_model, load_pretrained_audio_model

# Configure logging
logger = logging.getLogger(__name__)

def create_late_fusion_model(video_model, audio_model, num_classes=2):
    """
    Create a late fusion model that combines predictions from video and audio models.
    
    Args:
        video_model: Pretrained video model
        audio_model: Pretrained audio model
        num_classes: Number of output classes
        
    Returns:
        model: Compiled fusion model
    """
    logger.info("Creating late fusion model...")
    
    # Get the outputs of the models before the final dense layer
    video_features = video_model.layers[-2].output
    audio_features = audio_model.layers[-2].output
    
    # Create new input layers
    video_input = video_model.input
    audio_input = audio_model.input
    
    # Concatenate features
    concat_features = layers.Concatenate()([video_features, audio_features])
    
    # Classification head
    x = layers.Dense(256, activation='relu')(concat_features)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create fusion model
    fusion_model = models.Model(
        inputs=[video_input, audio_input],
        outputs=outputs
    )
    
    # Compile model
    fusion_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Late fusion model created")
    
    return fusion_model

def create_early_fusion_model(video_shape=(64, 224, 224, 3), audio_shape=(128, 87, 1), num_classes=2):
    """
    Create an early fusion model that processes both video and audio data.
    
    Args:
        video_shape: Shape of input videos
        audio_shape: Shape of input spectrograms
        num_classes: Number of output classes
        
    Returns:
        model: Compiled fusion model
    """
    logger.info("Creating early fusion model...")
    
    # Create video branch
    video_input = layers.Input(shape=video_shape)
    
    # Simple 3D CNN for video processing
    video_x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(video_input)
    video_x = layers.MaxPooling3D(pool_size=(1, 2, 2))(video_x)
    video_x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(video_x)
    video_x = layers.MaxPooling3D(pool_size=(1, 2, 2))(video_x)
    video_x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu')(video_x)
    video_x = layers.MaxPooling3D(pool_size=(2, 2, 2))(video_x)
    
    # Flatten video features
    video_x = layers.GlobalAveragePooling3D()(video_x)
    video_x = layers.Dense(512, activation='relu')(video_x)
    
    # Create audio branch
    audio_input = layers.Input(shape=audio_shape)
    
    # Simple 2D CNN for audio processing
    audio_x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(audio_input)
    audio_x = layers.MaxPooling2D(pool_size=(2, 2))(audio_x)
    audio_x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(audio_x)
    audio_x = layers.MaxPooling2D(pool_size=(2, 2))(audio_x)
    audio_x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(audio_x)
    audio_x = layers.MaxPooling2D(pool_size=(2, 2))(audio_x)
    
    # Flatten audio features
    audio_x = layers.GlobalAveragePooling2D()(audio_x)
    audio_x = layers.Dense(512, activation='relu')(audio_x)
    
    # Concatenate features
    concat_features = layers.Concatenate()([video_x, audio_x])
    
    # Classification head
    x = layers.Dense(512, activation='relu')(concat_features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create fusion model
    fusion_model = models.Model(
        inputs=[video_input, audio_input],
        outputs=outputs
    )
    
    # Compile model with focal loss for handling class imbalance
    focal_loss = tfa.losses.SigmoidFocalCrossEntropy(
        alpha=0.75,  # Focus more on fraud class
        gamma=2.0    # Focus more on hard examples
    )
    
    fusion_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tfa.metrics.F1Score(num_classes=num_classes, average='macro')
        ]
    )
    
    logger.info("Early fusion model created")
    
    return fusion_model

def create_multistream_transformer_fusion(video_shape=(64, 224, 224, 3), audio_shape=(128, 87, 1), num_classes=2):
    """
    Create a multistream transformer-based fusion model that uses attention
    to combine video and audio features.
    
    Args:
        video_shape: Shape of input videos
        audio_shape: Shape of input spectrograms
        num_classes: Number of output classes
        
    Returns:
        model: Compiled fusion model
    """
    logger.info("Creating multistream transformer fusion model...")
    
    # Create video branch with a simplified 3D CNN
    video_input = layers.Input(shape=video_shape)
    
    video_x = layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu')(video_input)
    video_x = layers.MaxPooling3D(pool_size=(1, 2, 2))(video_x)
    video_x = layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu')(video_x)
    video_x = layers.MaxPooling3D(pool_size=(1, 2, 2))(video_x)
    video_x = layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu')(video_x)
    video_x = layers.MaxPooling3D(pool_size=(2, 2, 2))(video_x)
    
    # Reshape to sequence for transformer
    video_x = layers.Reshape((-1, 128))(video_x)
    
    # Create audio branch with a simplified 2D CNN
    audio_input = layers.Input(shape=audio_shape)
    
    audio_x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(audio_input)
    audio_x = layers.MaxPooling2D(pool_size=(2, 2))(audio_x)
    audio_x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(audio_x)
    audio_x = layers.MaxPooling2D(pool_size=(2, 2))(audio_x)
    audio_x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(audio_x)
    audio_x = layers.MaxPooling2D(pool_size=(2, 2))(audio_x)
    
    # Reshape to sequence for transformer
    audio_x = layers.Reshape((-1, 128))(audio_x)
    
    # Cross-modal attention (video attending to audio)
    video_audio_attn = layers.MultiHeadAttention(
        num_heads=8, key_dim=64
    )(video_x, audio_x)
    video_x = layers.Add()([video_x, video_audio_attn])
    video_x = layers.LayerNormalization(epsilon=1e-6)(video_x)
    
    # Cross-modal attention (audio attending to video)
    audio_video_attn = layers.MultiHeadAttention(
        num_heads=8, key_dim=64
    )(audio_x, video_x)
    audio_x = layers.Add()([audio_x, audio_video_attn])
    audio_x = layers.LayerNormalization(epsilon=1e-6)(audio_x)
    
    # Global pooling
    video_x = layers.GlobalAveragePooling1D()(video_x)
    audio_x = layers.GlobalAveragePooling1D()(audio_x)
    
    # Concatenate features
    concat_features = layers.Concatenate()([video_x, audio_x])
    
    # Classification head
    x = layers.Dense(512, activation='relu')(concat_features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create fusion model
    fusion_model = models.Model(
        inputs=[video_input, audio_input],
        outputs=outputs
    )
    
    # Compile model with focal loss for handling class imbalance
    focal_loss = tfa.losses.SigmoidFocalCrossEntropy(
        alpha=0.75,  # Focus more on fraud class
        gamma=2.0    # Focus more on hard examples
    )
    
    fusion_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=focal_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tfa.metrics.F1Score(num_classes=num_classes, average='macro')
        ]
    )
    
    logger.info("Multistream transformer fusion model created")
    
    return fusion_model

def create_weighted_ensemble(video_model, audio_model, video_weight=0.6, audio_weight=0.4):
    """
    Create a weighted ensemble that combines predictions from video and audio models.
    This is a simple approach that doesn't require training a fusion model.
    
    Args:
        video_model: Pretrained video model
        audio_model: Pretrained audio model
        video_weight: Weight for video predictions
        audio_weight: Weight for audio predictions
        
    Returns:
        ensemble_function: Function that takes video and audio inputs and returns weighted predictions
    """
    logger.info(f"Creating weighted ensemble with weights: video={video_weight}, audio={audio_weight}")
    
    def ensemble_predict(video_input, audio_input):
        """
        Make predictions using weighted ensemble.
        
        Args:
            video_input: Input video
            audio_input: Input audio
            
        Returns:
            predictions: Weighted predictions
        """
        # Get predictions from individual models
        video_pred = video_model.predict(video_input)
        audio_pred = audio_model.predict(audio_input)
        
        # Apply weights
        weighted_pred = video_weight * video_pred + audio_weight * audio_pred
        
        return weighted_pred
    
    return ensemble_predict

def create_adaptive_fusion_model(video_model, audio_model, num_classes=2):
    """
    Create an adaptive fusion model that learns to weight video and audio predictions
    based on input quality.
    
    Args:
        video_model: Pretrained video model
        audio_model: Pretrained audio model
        num_classes: Number of output classes
        
    Returns:
        model: Compiled adaptive fusion model
    """
    logger.info("Creating adaptive fusion model...")
    
    # Create new input layers
    video_input = video_model.input
    audio_input = audio_model.input
    
    # Get features from individual models
    video_features = video_model.layers[-2].output
    audio_features = audio_model.layers[-2].output
    
    # Quality assessment network
    video_quality = layers.Dense(64, activation='relu')(video_features)
    video_quality = layers.Dense(1, activation='sigmoid', name='video_quality')(video_quality)
    
    audio_quality = layers.Dense(64, activation='relu')(audio_features)
    audio_quality = layers.Dense(1, activation='sigmoid', name='audio_quality')(audio_quality)
    
    # Normalize weights to sum to 1
    quality_concat = layers.Concatenate()([video_quality, audio_quality])
    normalized_weights = layers.Softmax()(quality_concat)
    
    # Extract individual weights
    video_weight = layers.Lambda(lambda x: x[:, 0:1])(normalized_weights)
    audio_weight = layers.Lambda(lambda x: x[:, 1:2])(normalized_weights)
    
    # Get predictions from individual models
    video_pred = video_model.layers[-1].output
    audio_pred = audio_model.layers[-1].output
    
    # Apply weights
    weighted_video = layers.Multiply()([video_pred, video_weight])
    weighted_audio = layers.Multiply()([audio_pred, audio_weight])
    
    # Combine predictions
    weighted_sum = layers.Add()([weighted_video, weighted_audio])
    
    # Create adaptive fusion model
    fusion_model = models.Model(
        inputs=[video_input, audio_input],
        outputs=weighted_sum
    )
    
    # Compile model
    fusion_model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Adaptive fusion model created")
    
    return fusion_model

def load_pretrained_fusion_model(model_path, custom_objects=None):
    """
    Load a pretrained fusion model.
    
    Args:
        model_path: Path to model weights
        custom_objects: Custom objects to load
        
    Returns:
        model: Loaded model
    """
    logger.info(f"Loading pretrained fusion model from {model_path}")
    
    # If custom objects not provided, use default ones
    if custom_objects is None:
        custom_objects = {
            'F1Score': tfa.metrics.F1Score,
            'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy
        }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("Fusion model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading fusion model: {e}")
        return None

def create_fusion_dataset(video_dataset, audio_dataset):
    """
    Create a combined dataset for fusion model training.
    
    Args:
        video_dataset: TensorFlow dataset with video data
        audio_dataset: TensorFlow dataset with audio data
        
    Returns:
        fusion_dataset: Combined dataset
    """
    logger.info("Creating fusion dataset...")
    
    # Extract elements from individual datasets
    video_data, labels = zip(*[(video, label) for video, label in video_dataset])
    audio_data = [audio for audio, _ in audio_dataset]
    
    # Create fusion dataset
    fusion_dataset = tf.data.Dataset.from_tensor_slices(
        ((video_data, audio_data), labels)
    )
    
    return fusion_dataset

def evaluate_fusion_model(fusion_model, video_test_dataset, audio_test_dataset):
    """
    Evaluate a fusion model.
    
    Args:
        fusion_model: Fusion model to evaluate
        video_test_dataset: Test dataset with video data
        audio_test_dataset: Test dataset with audio data
        
    Returns:
        results: Evaluation results
    """
    logger.info("Evaluating fusion model...")
    
    # Create fusion test dataset
    fusion_test_dataset = create_fusion_dataset(video_test_dataset, audio_test_dataset)
    
    # Evaluate model
    results = fusion_model.evaluate(fusion_test_dataset)
    
    # Print results
    metrics = fusion_model.metrics_names
    for metric, value in zip(metrics, results):
        logger.info(f"{metric}: {value}")
    
    return dict(zip(metrics, results))

def save_fusion_model(model, model_path):
    """
    Save a fusion model.
    
    Args:
        model: Fusion model to save
        model_path: Path to save the model
        
    Returns:
        None
    """
    logger.info(f"Saving fusion model to {model_path}")
    
    try:
        model.save(model_path)
        logger.info("Fusion model saved successfully")
    except Exception as e:
        logger.error(f"Error saving fusion model: {e}")