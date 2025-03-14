import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Reshape, MultiHeadAttention, LayerNormalization, Layer, Add,
    Flatten, BatchNormalization, Activation
)
import tensorflow_addons as tfa
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class PatchExtractor(Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dim])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

def mlp(x, hidden_units, dropout_rate=0.1):
    """MLP for transformer."""
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def create_ast_model(input_shape=(128, 87, 1), num_classes=2):
    """
    Create an Audio Spectrogram Transformer (AST) model.
    
    Args:
        input_shape: Shape of input spectrograms (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled AST model
    """
    logger.info("Creating AST model...")
    
    # Parameters
    patch_size = 16  # Size of patches to extract from the input image
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    projection_dim = 256  # Embedding dimension
    num_heads = 8  # Number of attention heads
    transformer_units = [  # Size of the transformer layers
        projection_dim * 2,
        projection_dim,
    ]
    transformer_layers = 8  # Number of transformer layers
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    
    # Reshape to ensure 4D inputs for patch extraction
    x = layers.Reshape((input_shape[0], input_shape[1], input_shape[2]))(inputs)
    
    # Create patches
    patches = PatchExtractor(patch_size)(x)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Create transformer encoder
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads
        )(x1, x1)
        
        # Skip connection 1
        x2 = Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, transformer_units)
        
        # Skip connection 2
        encoded_patches = Add()([x3, x2])
    
    # Create a [batch_size, projection_dim] tensor
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    
    # Classification head
    x = Dense(512, activation="relu")(representation)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
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
    
    logger.info(f"AST model created with input shape {input_shape}")
    
    return model

def create_panns_model(input_shape=(128, 87, 1), num_classes=2):
    """
    Create a PANNs-inspired model for audio classification.
    PANNs (Pre-trained Audio Neural Networks) are state-of-the-art audio classifiers.
    
    Args:
        input_shape: Shape of input spectrograms (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled PANNs model
    """
    logger.info("Creating PANNs model...")
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Block 2
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Block 3
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Block 4
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Global pooling
    x = GlobalAveragePooling2D()(x)
    
    # Classification head
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"PANNs model created with input shape {input_shape}")
    
    return model

def create_wav2vec_model(input_shape=(128, 87, 1), num_classes=2):
    """
    Create a Wav2Vec-inspired model for audio classification.
    
    Args:
        input_shape: Shape of input spectrograms (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled Wav2Vec model
    """
    logger.info("Creating Wav2Vec model...")
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    
    # CNN encoder
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Reshape for LSTM
    x = Reshape((-1, x.shape[-1] * x.shape[-2]))(x)
    
    # Bidirectional LSTM layers for temporal modeling
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    
    # Classification head
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
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
    
    logger.info(f"Wav2Vec model created with input shape {input_shape}")
    
    return model

def create_cnn_model(input_shape=(128, 87, 1), num_classes=2):
    """
    Create a simple CNN model for audio classification.
    
    Args:
        input_shape: Shape of input spectrograms (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        model: Compiled CNN model
    """
    logger.info("Creating CNN model...")
    
    # Create model
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"CNN model created with input shape {input_shape}")
    
    return model

def create_multifeature_model(input_shape=(128, 87, 1), num_additional_features=20, num_classes=2):
    """
    Create a model that uses both spectrogram and additional audio features.
    
    Args:
        input_shape: Shape of input spectrograms (height, width, channels)
        num_additional_features: Number of additional features
        num_classes: Number of output classes
        
    Returns:
        model: Compiled multi-feature model
    """
    logger.info("Creating multi-feature model...")
    
    # Create inputs
    spectrogram_input = layers.Input(shape=input_shape)
    additional_features_input = layers.Input(shape=(num_additional_features,))
    
    # Process spectrogram with CNN
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(spectrogram_input)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    spectrogram_features = Dense(256, activation='relu')(x)
    
    # Process additional features
    additional_features = Dense(64, activation='relu')(additional_features_input)
    additional_features = Dense(128, activation='relu')(additional_features)
    
    # Concatenate features
    combined_features = layers.Concatenate()([spectrogram_features, additional_features])
    
    # Classification head
    x = Dense(256, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(
        inputs=[spectrogram_input, additional_features_input],
        outputs=outputs
    )
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Multi-feature model created with input shapes {input_shape} and {(num_additional_features,)}")
    
    return model

def load_pretrained_audio_model(model_path, custom_objects=None):
    """
    Load a pretrained audio model.
    
    Args:
        model_path: Path to model weights
        custom_objects: Custom objects to load
        
    Returns:
        model: Loaded model
    """
    logger.info(f"Loading pretrained audio model from {model_path}")
    
    # If custom objects not provided, use default ones
    if custom_objects is None:
        custom_objects = {
            'PatchExtractor': PatchExtractor,
            'PatchEncoder': PatchEncoder,
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

def fine_tune_audio_model(model, train_dataset, val_dataset, epochs=10, learning_rate=1e-5):
    """
    Fine-tune a pretrained audio model.
    
    Args:
        model: Pretrained model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of epochs for fine-tuning
        learning_rate: Learning rate for fine-tuning
        
    Returns:
        model: Fine-tuned model
        history: Training history
    """
    logger.info("Fine-tuning audio model...")
    
    # Compile model with a lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.75, gamma=2.0),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tfa.metrics.F1Score(num_classes=model.output.shape[-1], average='macro')
        ]
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            monitor='val_loss'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"logs/fine_tune_{model.name}_{tf.timestamp()}",
            histogram_freq=1
        )
    ]
    
    # Fine-tune model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    logger.info("Fine-tuning completed")
    
    return model, history