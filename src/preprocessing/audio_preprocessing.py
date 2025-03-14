import librosa
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import random
from scipy import signal
import soundfile as sf

# Configure logging
logger = logging.getLogger(__name__)

def extract_audio_features(audio_path, sr=22050, duration=5, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract mel spectrogram features from audio file
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        duration: Duration in seconds to load
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        mel_spectrogram: Mel spectrogram features
    """
    try:
        # Load audio file with fixed duration
        y, _ = librosa.load(audio_path, sr=sr, duration=duration, res_type='kaiser_fast')
        
        # Handle audio files shorter than duration
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        
        # Extract mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        
        # Convert to dB scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Normalize to [0, 1] range
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min() + 1e-8)
        
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        # Return empty spectrogram in case of error
        mel_spectrogram = np.zeros((n_mels, int(np.ceil(sr * duration / hop_length))))
    
    return mel_spectrogram

def extract_additional_features(audio_path, sr=22050, duration=5):
    """
    Extract additional audio features that might be relevant for fraud detection
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        duration: Duration in seconds to load
        
    Returns:
        features: Dictionary of additional features
    """
    try:
        # Load audio
        y, _ = librosa.load(audio_path, sr=sr, duration=duration, res_type='kaiser_fast')
        
        # Handle audio files shorter than duration
        if len(y) < sr * duration:
            y = np.pad(y, (0, sr * duration - len(y)))
        
        # Temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        
        # Rhythmic features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # MFCC (capture timbre characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        
        # RMS energy
        rms = librosa.feature.rms(y=y).mean()
        
        features = {
            'zero_crossing_rate': zero_crossing_rate,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'tempo': tempo,
            'mfccs': mfccs,
            'rms': rms
        }
        
    except Exception as e:
        logger.error(f"Error extracting additional features from {audio_path}: {e}")
        # Return empty features in case of error
        features = {
            'zero_crossing_rate': 0,
            'spectral_centroid': 0,
            'spectral_bandwidth': 0,
            'spectral_rolloff': 0,
            'tempo': 0,
            'mfccs': np.zeros(13),
            'rms': 0
        }
    
    return features

def create_audio_dataset(audio_dir, sr=22050, duration=5, n_mels=128, n_fft=2048, hop_length=512,
                        batch_size=32, val_split=0.2, test_split=0.1, random_state=42, balance_classes=True):
    """
    Create a dataset from audio files for fraud detection
    
    Args:
        audio_dir: Directory containing 'Fraud' and 'Legit' subfolders with audio files
        sr: Sample rate
        duration: Duration in seconds to load
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        batch_size: Batch size for training
        val_split: Validation split ratio
        test_split: Test split ratio
        random_state: Random seed for reproducibility
        balance_classes: Whether to balance classes
        
    Returns:
        train_dataset, val_dataset, test_dataset: TensorFlow datasets for training, validation, and testing
    """
    # Create lists to store audio paths and labels
    audio_paths = []
    labels = []
    
    # Get paths for fraud audio
    fraud_dir = os.path.join(audio_dir, 'Fraud')
    for audio_file in os.listdir(fraud_dir):
        if audio_file.endswith(('.wav', '.mp3', '.ogg')):
            audio_paths.append(os.path.join(fraud_dir, audio_file))
            labels.append(1)  # 1 for fraud
    
    # Get paths for legitimate audio
    legit_dir = os.path.join(audio_dir, 'Legit')
    for audio_file in os.listdir(legit_dir):
        if audio_file.endswith(('.wav', '.mp3', '.ogg')):
            audio_paths.append(os.path.join(legit_dir, audio_file))
            labels.append(0)  # 0 for legitimate
    
    # Balance classes if needed
    if balance_classes:
        audio_paths, labels = balance_dataset(audio_paths, labels)
    
    # Split data into train, validation, and test sets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        audio_paths, labels, test_size=(val_split + test_split), 
        random_state=random_state, stratify=labels
    )
    
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1-val_ratio),
        random_state=random_state, stratify=temp_labels
    )
    
    logger.info(f"Audio dataset split: {len(train_paths)} train, {len(val_paths)} validation, {len(test_paths)} test")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    
    # Preprocess datasets
    train_dataset = train_dataset.map(
        lambda x, y: tf.py_function(
            preprocess_audio, 
            [x, y, sr, duration, n_mels, n_fft, hop_length, True], 
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    val_dataset = val_dataset.map(
        lambda x, y: tf.py_function(
            preprocess_audio, 
            [x, y, sr, duration, n_mels, n_fft, hop_length, False], 
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    test_dataset = test_dataset.map(
        lambda x, y: tf.py_function(
            preprocess_audio, 
            [x, y, sr, duration, n_mels, n_fft, hop_length, False], 
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Cache, batch, and prefetch datasets
    train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def preprocess_audio(audio_path, label, sr, duration, n_mels, n_fft, hop_length, augment):
    """
    Preprocess audio for the model.
    
    Args:
        audio_path: Path to audio file
        label: Audio label
        sr: Sample rate
        duration: Duration in seconds to load
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        augment: Whether to apply augmentation
        
    Returns:
        mel_spectrogram: Preprocessed mel spectrogram
        label: Audio label
    """
    try:
        # Extract mel spectrogram
        mel_spectrogram = extract_audio_features(
            audio_path.numpy().decode(), 
            sr=sr, duration=duration, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        
        # Apply augmentation if needed
        if augment:
            mel_spectrogram = augment_spectrogram(mel_spectrogram)
        
        # Add channel dimension for CNN (height, width, channels)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
        
        return mel_spectrogram.astype(np.float32), label
    
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        # Return an empty spectrogram in case of error
        empty_spec = np.zeros((n_mels, int(np.ceil(sr * duration / hop_length)), 1), dtype=np.float32)
        return empty_spec, label

def augment_spectrogram(mel_spectrogram):
    """
    Apply data augmentation to mel spectrogram.
    
    Args:
        mel_spectrogram: Mel spectrogram to augment
        
    Returns:
        mel_spectrogram: Augmented mel spectrogram
    """
    # Add random noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.01, mel_spectrogram.shape)
        mel_spectrogram = mel_spectrogram + noise
        mel_spectrogram = np.clip(mel_spectrogram, 0, 1)
    
    # Frequency masking
    if random.random() > 0.5:
        freq_mask_param = random.randint(1, 20)
        mask_begin = random.randint(0, mel_spectrogram.shape[0] - freq_mask_param)
        mel_spectrogram[mask_begin:mask_begin + freq_mask_param, :] = 0
    
    # Time masking
    if random.random() > 0.5:
        time_mask_param = random.randint(1, 20)
        mask_begin = random.randint(0, mel_spectrogram.shape[1] - time_mask_param)
        mel_spectrogram[:, mask_begin:mask_begin + time_mask_param] = 0
    
    # SpecAugment-like augmentation
    if random.random() > 0.5:
        # Time stretching (simulated by warping the spectrogram)
        warp_factor = random.uniform(0.8, 1.2)
        mel_spectrogram = np.array([signal.resample(row, int(len(row) * warp_factor)) for row in mel_spectrogram])
        
        # Ensure correct shape
        if mel_spectrogram.shape[1] < mel_spectrogram.shape[1]:
            # Pad if too short
            mel_spectrogram = np.pad(
                mel_spectrogram, 
                ((0, 0), (0, mel_spectrogram.shape[1] - mel_spectrogram.shape[1]))
            )
        elif mel_spectrogram.shape[1] > mel_spectrogram.shape[1]:
            # Trim if too long
            mel_spectrogram = mel_spectrogram[:, :mel_spectrogram.shape[1]]
    
    return mel_spectrogram

def augment_audio_files(audio_dir, output_dir, augmentations_per_file=2, sr=22050):
    """
    Generate augmented audio files to increase dataset size.
    
    Args:
        audio_dir: Directory containing 'Fraud' and 'Legit' subfolders with audio files
        output_dir: Directory to save augmented audio files
        augmentations_per_file: Number of augmentations to generate per file
        sr: Sample rate
    """
    # Create output directories
    fraud_output_dir = os.path.join(output_dir, 'Fraud')
    legit_output_dir = os.path.join(output_dir, 'Legit')
    os.makedirs(fraud_output_dir, exist_ok=True)
    os.makedirs(legit_output_dir, exist_ok=True)
    
    # Process fraud audio files
    fraud_dir = os.path.join(audio_dir, 'Fraud')
    for audio_file in tqdm(os.listdir(fraud_dir), desc="Augmenting fraud audio"):
        if audio_file.endswith(('.wav', '.mp3', '.ogg')):
            audio_path = os.path.join(fraud_dir, audio_file)
            
            # Load audio
            try:
                y, _ = librosa.load(audio_path, sr=sr, res_type='kaiser_fast')
                
                # Create augmented versions
                for i in range(augmentations_per_file):
                    # Apply augmentations
                    augmented_y = y.copy()
                    
                    # Time stretching
                    if random.random() > 0.5:
                        stretch_factor = random.uniform(0.8, 1.2)
                        augmented_y = librosa.effects.time_stretch(augmented_y, rate=stretch_factor)
                    
                    # Pitch shifting
                    if random.random() > 0.5:
                        n_steps = random.uniform(-3, 3)
                        augmented_y = librosa.effects.pitch_shift(augmented_y, sr=sr, n_steps=n_steps)
                    
                    # Add noise
                    if random.random() > 0.5:
                        noise_factor = random.uniform(0.001, 0.01)
                        noise = np.random.normal(0, noise_factor, len(augmented_y))
                        augmented_y = augmented_y + noise
                    
                    # Save augmented audio
                    output_filename = f"{os.path.splitext(audio_file)[0]}_aug_{i}{os.path.splitext(audio_file)[1]}"
                    output_path = os.path.join(fraud_output_dir, output_filename)
                    sf.write(output_path, augmented_y, sr)
            
            except Exception as e:
                logger.error(f"Error augmenting {audio_path}: {e}")
    
    # Process legitimate audio files
    legit_dir = os.path.join(audio_dir, 'Legit')
    for audio_file in tqdm(os.listdir(legit_dir), desc="Augmenting legitimate audio"):
        if audio_file.endswith(('.wav', '.mp3', '.ogg')):
            audio_path = os.path.join(legit_dir, audio_file)
            
            # Load audio
            try:
                y, _ = librosa.load(audio_path, sr=sr, res_type='kaiser_fast')
                
                # Create augmented versions
                for i in range(augmentations_per_file):
                    # Apply augmentations
                    augmented_y = y.copy()
                    
                    # Time stretching
                    if random.random() > 0.5:
                        stretch_factor = random.uniform(0.8, 1.2)
                        augmented_y = librosa.effects.time_stretch(augmented_y, rate=stretch_factor)
                    
                    # Pitch shifting
                    if random.random() > 0.5:
                        n_steps = random.uniform(-3, 3)
                        augmented_y = librosa.effects.pitch_shift(augmented_y, sr=sr, n_steps=n_steps)
                    
                    # Add noise
                    if random.random() > 0.5:
                        noise_factor = random.uniform(0.001, 0.01)
                        noise = np.random.normal(0, noise_factor, len(augmented_y))
                        augmented_y = augmented_y + noise
                    
                    # Save augmented audio
                    output_filename = f"{os.path.splitext(audio_file)[0]}_aug_{i}{os.path.splitext(audio_file)[1]}"
                    output_path = os.path.join(legit_output_dir, output_filename)
                    sf.write(output_path, augmented_y, sr)
            
            except Exception as e:
                logger.error(f"Error augmenting {audio_path}: {e}")
    
    logger.info(f"Augmented audio files saved to {output_dir}")

def balance_dataset(audio_paths, labels):
    """
    Balance the dataset by oversampling the minority class.
    
    Args:
        audio_paths: List of audio paths
        labels: List of labels
        
    Returns:
        balanced_paths, balanced_labels: Balanced dataset
    """
    # Count occurrences of each class
    fraud_indices = [i for i, label in enumerate(labels) if label == 1]
    legit_indices = [i for i, label in enumerate(labels) if label == 0]
    
    # Determine minority class
    if len(fraud_indices) < len(legit_indices):
        minority_indices = fraud_indices
        majority_indices = legit_indices
    else:
        minority_indices = legit_indices
        majority_indices = fraud_indices
    
    # Oversample minority class
    oversampled_indices = np.random.choice(minority_indices, 
                                          size=len(majority_indices) - len(minority_indices), 
                                          replace=True)
    
    # Combine all indices
    all_indices = list(range(len(labels))) + list(oversampled_indices)
    
    # Create balanced dataset
    balanced_paths = [audio_paths[i] for i in all_indices]
    balanced_labels = [labels[i] for i in all_indices]
    
    logger.info(f"Balanced audio dataset: {len(balanced_paths)} audio files, "
               f"{sum(balanced_labels)} fraud, {len(balanced_labels) - sum(balanced_labels)} legitimate")
    
    return balanced_paths, balanced_labels

def create_mfcc_dataset(audio_dir, sr=22050, duration=5, n_mfcc=40, batch_size=32):
    """
    Create a dataset with MFCC features.
    
    Args:
        audio_dir: Directory containing 'Fraud' and 'Legit' subfolders with audio files
        sr: Sample rate
        duration: Duration in seconds to load
        n_mfcc: Number of MFCC coefficients
        batch_size: Batch size for training
        
    Returns:
        train_dataset, val_dataset, test_dataset: TensorFlow datasets for training, validation, and testing
    """
    # Create lists to store MFCC features and labels
    mfcc_features = []
    labels = []
    
    # Process fraud audio files
    fraud_dir = os.path.join(audio_dir, 'Fraud')
    for audio_file in tqdm(os.listdir(fraud_dir), desc="Processing fraud audio"):
        if audio_file.endswith(('.wav', '.mp3', '.ogg')):
            audio_path = os.path.join(fraud_dir, audio_file)
            
            try:
                # Load audio
                y, _ = librosa.load(audio_path, sr=sr, duration=duration, res_type='kaiser_fast')
                
                # Handle audio files shorter than duration
                if len(y) < sr * duration:
                    y = np.pad(y, (0, sr * duration - len(y)))
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                
                # Normalize features
                mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
                
                mfcc_features.append(mfcc)
                labels.append(1)  # 1 for fraud
            
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
    
    # Process legitimate audio files
    legit_dir = os.path.join(audio_dir, 'Legit')
    for audio_file in tqdm(os.listdir(legit_dir), desc="Processing legitimate audio"):
        if audio_file.endswith(('.wav', '.mp3', '.ogg')):
            audio_path = os.path.join(legit_dir, audio_file)
            
            try:
                # Load audio
                y, _ = librosa.load(audio_path, sr=sr, duration=duration, res_type='kaiser_fast')
                
                # Handle audio files shorter than duration
                if len(y) < sr * duration:
                    y = np.pad(y, (0, sr * duration - len(y)))
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                
                # Normalize features
                mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
                
                mfcc_features.append(mfcc)
                labels.append(0)  # 0 for legitimate
            
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
    
    # Convert to numpy arrays
    mfcc_features = np.array(mfcc_features)
    labels = np.array(labels)
    
    # Add channel dimension for CNN
    mfcc_features = np.expand_dims(mfcc_features, axis=-1)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        mfcc_features, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # Apply data augmentation to training dataset
    train_dataset = train_dataset.map(
        lambda x, y: (tf.py_function(augment_mfcc, [x], tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def augment_mfcc(mfcc):
    """
    Apply data augmentation to MFCC features.
    
    Args:
        mfcc: MFCC features to augment
        
    Returns:
        mfcc: Augmented MFCC features
    """
    # Add random noise
    if random.random() > 0.5:
        noise = tf.random.normal(shape=tf.shape(mfcc), mean=0.0, stddev=0.01)
        mfcc = mfcc + noise
    
    # Frequency masking
    if random.random() > 0.5:
        freq_mask_param = tf.random.uniform([], minval=1, maxval=10, dtype=tf.int32)
        mfcc = tf.tensor_scatter_nd_update(
            mfcc,
            indices=tf.range(freq_mask_param)[:, tf.newaxis],
            updates=tf.zeros([freq_mask_param, tf.shape(mfcc)[1], 1])
        )
    
    # Time masking
    if random.random() > 0.5:
        time_mask_param = tf.random.uniform([], minval=1, maxval=10, dtype=tf.int32)
        mfcc = tf.tensor_scatter_nd_update(
            mfcc,
            indices=tf.range(time_mask_param)[:, tf.newaxis],
            updates=tf.zeros([time_mask_param, tf.shape(mfcc)[0], 1])
        )
    
    return mfcc