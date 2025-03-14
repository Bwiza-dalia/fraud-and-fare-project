import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling3D, Conv3D, MaxPooling3D, 
    Dropout, LSTM, TimeDistributed, Layer, MultiHeadAttention,
    LayerNormalization, Add
)
import tensorflow_addons as tfa
import logging

# Configure logging
logger = logging.getLogger(__name__)

class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):
        # Create positional indices
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        # Get positional embeddings
        embedded_positions = self.position_embedding(positions)
        # Add positional embeddings to inputs
        return inputs + embedded_positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'output_dim': self.output_dim,
        })
        return config

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

def create_frame_feature_extractor(input_shape=(224, 224, 3), trainable=False):
    """
    Create a frame feature extractor using EfficientNetB0.
    
    Args:
        input_shape: Input shape of a single frame
        trainable: Whether to train the feature extractor
        
    Returns:
        model: Frame feature extractor model
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze the feature extractor
    base_model.trainable = trainable
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    features = base_model(inputs)
    outputs = layers.Dense(512, activation='relu')(features)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def create_vivit_model(input_shape=(64, 224, 224, 3), num_classes=2):
    """
    Create a ViViT (Video Vision Transformer) model for fraud detection.
    
    Args:
        input_shape: Shape of input videos (frames, height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled ViViT model
    """
    logger.info("Creating ViViT model...")
    
    # Parameters
    embed_dim = 512
    num_heads = 8
    ff_dim = 1024
    num_transformer_blocks = 6
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    
    # Extract frame features
    frame_extractor = create_frame_feature_extractor(input_shape=input_shape[1:])
    
    # Apply frame extractor to each frame
    x = TimeDistributed(frame_extractor)(inputs)
    
    # Add positional embedding
    x = PositionalEmbedding(input_shape[0], embed_dim)(x)
    
    # Apply transformer blocks
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with focal loss for handling class imbalance
    focal_loss = tfa.losses.SigmoidFocalCrossEntropy(
        alpha=0.75,  # Focus more on fraud class
        gamma=2.0    # Focus more on hard examples
    )
    
    model.compile(
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
    
    logger.info(f"ViViT model created with input shape {input_shape}")
    
    return model

def create_slowfast_model(input_shape=(64, 224, 224, 3), num_classes=2):
    """
    Create a SlowFast network for fraud detection.
    This model uses two pathways: a slow pathway for spatial information and
    a fast pathway for motion information.
    
    Args:
        input_shape: Shape of input videos (frames, height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled SlowFast model
    """
    logger.info("Creating SlowFast model...")
    
    # Parameters
    alpha = 8  # Frame rate ratio between fast and slow pathways
    beta = 1/8  # Channel ratio between fast and slow pathways
    
    # Input
    inputs = layers.Input(shape=input_shape)
    
    # Create slow pathway (uses strided sampling)
    slow_frames = input_shape[0] // alpha
    indices = tf.linspace(0, input_shape[0]-1, slow_frames)
    indices = tf.cast(indices, dtype=tf.int32)
    slow_pathway = tf.gather(inputs, indices, axis=1)
    
    # Create fast pathway (uses all frames)
    fast_pathway = inputs
    
    # Slow pathway convolutional blocks
    slow = Conv3D(64, kernel_size=(1, 7, 7), strides=(1, 2, 2), padding='same')(slow_pathway)
    slow = layers.BatchNormalization()(slow)
    slow = layers.Activation('relu')(slow)
    slow = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(slow)
    
    # Slow pathway residual blocks
    slow = slow_residual_block(slow, 64, 256)
    slow = slow_residual_block(slow, 128, 512, strides=(2, 2, 2))
    slow = slow_residual_block(slow, 256, 1024, strides=(2, 2, 2))
    slow = slow_residual_block(slow, 512, 2048, strides=(2, 2, 2))
    
    # Fast pathway convolutional blocks (fewer channels)
    fast_channels = int(64 * beta)
    fast = Conv3D(fast_channels, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding='same')(fast_pathway)
    fast = layers.BatchNormalization()(fast)
    fast = layers.Activation('relu')(fast)
    fast = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(fast)
    
    # Fast pathway residual blocks
    fast = fast_residual_block(fast, int(64 * beta), int(256 * beta))
    fast = fast_residual_block(fast, int(128 * beta), int(512 * beta), strides=(2, 2, 2))
    fast = fast_residual_block(fast, int(256 * beta), int(1024 * beta), strides=(2, 2, 2))
    fast = fast_residual_block(fast, int(512 * beta), int(2048 * beta), strides=(2, 2, 2))
    
    # Global pooling
    slow = GlobalAveragePooling3D()(slow)
    fast = GlobalAveragePooling3D()(fast)
    
    # Fusion
    fusion = layers.Concatenate()([slow, fast])
    
    # Classification head
    x = layers.Dense(512, activation='relu')(fusion)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"SlowFast model created with input shape {input_shape}")
    
    return model

def slow_residual_block(x, filters, filters_expansion, strides=(1, 1, 1)):
    """Create a residual block for the slow pathway."""
    shortcut = x
    
    # Determine if we need a projection shortcut
    if strides != (1, 1, 1) or x.shape[-1] != filters_expansion:
        shortcut = Conv3D(filters_expansion, kernel_size=1, strides=strides)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # First conv
    x = Conv3D(filters, kernel_size=(1, 3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second conv
    x = Conv3D(filters, kernel_size=(1, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Third conv
    x = Conv3D(filters_expansion, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def fast_residual_block(x, filters, filters_expansion, strides=(1, 1, 1)):
    """Create a residual block for the fast pathway."""
    shortcut = x
    
    # Determine if we need a projection shortcut
    if strides != (1, 1, 1) or x.shape[-1] != filters_expansion:
        shortcut = Conv3D(filters_expansion, kernel_size=1, strides=strides)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # First conv (note the 3D temporal convolution with kernel size 3)
    x = Conv3D(filters, kernel_size=(3, 3, 3), strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second conv
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Third conv
    x = Conv3D(filters_expansion, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def create_3d_resnet_model(input_shape=(64, 224, 224, 3), num_classes=2):
    """
    Create a 3D ResNet model for fraud detection.
    This is a lightweight alternative to the ViViT and SlowFast models.
    
    Args:
        input_shape: Shape of input videos (frames, height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled 3D ResNet model
    """
    logger.info("Creating 3D ResNet model...")
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = Conv3D(64, kernel_size=(3, 7, 7), strides=(1, 2, 2), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64, 256)
    x = residual_block(x, 128, 512, strides=(2, 2, 2))
    x = residual_block(x, 256, 1024, strides=(2, 2, 2))
    x = residual_block(x, 512, 2048, strides=(2, 2, 2))
    
    # Global pooling
    x = GlobalAveragePooling3D()(x)
    
    # Classification head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with focal loss for handling class imbalance
    focal_loss = tfa.losses.SigmoidFocalCrossEntropy(
        alpha=0.75,  # Focus more on fraud class
        gamma=2.0    # Focus more on hard examples
    )
    
    model.compile(
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
    
    logger.info(f"3D ResNet model created with input shape {input_shape}")
    
    return model

def residual_block(x, filters, filters_expansion, strides=(1, 1, 1)):
    """Create a residual block for 3D ResNet."""
    shortcut = x
    
    # Determine if we need a projection shortcut
    if strides != (1, 1, 1) or x.shape[-1] != filters_expansion:
        shortcut = Conv3D(filters_expansion, kernel_size=1, strides=strides)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # First conv
    x = Conv3D(filters, kernel_size=1, strides=1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second conv
    x = Conv3D(filters, kernel_size=3, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Third conv
    x = Conv3D(filters_expansion, kernel_size=1)(x)
    x = layers.BatchNormalization()(x)
    
    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def create_cnn_lstm_model(input_shape=(64, 224, 224, 3), num_classes=2):
    """
    Create a CNN-LSTM model for fraud detection.
    This model uses a CNN to extract spatial features from each frame
    and an LSTM to model temporal relationships.
    
    Args:
        input_shape: Shape of input videos (frames, height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled CNN-LSTM model
    """
    logger.info("Creating CNN-LSTM model...")
    
    # Create frame feature extractor
    frame_extractor = create_frame_feature_extractor(input_shape=input_shape[1:])
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    
    # Extract features from each frame
    x = TimeDistributed(frame_extractor)(inputs)
    
    # LSTM layers
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"CNN-LSTM model created with input shape {input_shape}")
    
    return model

def load_pretrained_video_model(model_path, custom_objects=None):
    """
    Load a pretrained video model.
    
    Args:
        model_path: Path to model weights
        custom_objects: Custom objects to load
        
    Returns:
        model: Loaded model
    """
    logger.info(f"Loading pretrained video model from {model_path}")
    
    # If custom objects not provided, use default ones
    if custom_objects is None:
        custom_objects = {
            'PositionalEmbedding': PositionalEmbedding,
            'TransformerBlock': TransformerBlock,
            'F1Score': tfa.metrics.F1Score,
            'SigmoidFocalCrossEntropy': tfa.losses.SigmoidFocalCrossEntropy
        }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None