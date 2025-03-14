import os
import shutil
import random
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)

def create_dataset_splits(data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        data_dir: Directory containing 'Fraud' and 'Legit' subfolders
        output_dir: Directory to save the split datasets
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        None
    """
    # Validate ratios
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    # Create subdirectories for classes
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, 'Fraud'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'Legit'), exist_ok=True)
    
    # Process each class
    for class_name in ['Fraud', 'Legit']:
        class_dir = os.path.join(data_dir, class_name)
        files = os.listdir(class_dir)
        
        # Shuffle files
        random.seed(random_seed)
        random.shuffle(files)
        
        # Calculate split indices
        train_end = int(len(files) * train_ratio)
        val_end = train_end + int(len(files) * val_ratio)
        
        # Split files
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        # Copy files to output directories
        for file in train_files:
            shutil.copy2(
                os.path.join(class_dir, file),
                os.path.join(train_dir, class_name, file)
            )
        
        for file in val_files:
            shutil.copy2(
                os.path.join(class_dir, file),
                os.path.join(val_dir, class_name, file)
            )
        
        for file in test_files:
            shutil.copy2(
                os.path.join(class_dir, file),
                os.path.join(test_dir, class_name, file)
            )
    
    # Log dataset statistics
    logger.info(f"Dataset split created in {output_dir}")
    logger.info(f"Train set: {len(os.listdir(os.path.join(train_dir, 'Fraud')))} fraud, {len(os.listdir(os.path.join(train_dir, 'Legit')))} legit")
    logger.info(f"Validation set: {len(os.listdir(os.path.join(val_dir, 'Fraud')))} fraud, {len(os.listdir(os.path.join(val_dir, 'Legit')))} legit")
    logger.info(f"Test set: {len(os.listdir(os.path.join(test_dir, 'Fraud')))} fraud, {len(os.listdir(os.path.join(test_dir, 'Legit')))} legit")

def analyze_dataset(data_dir, output_dir=None):
    """
    Analyze a dataset and generate statistics and visualizations.
    
    Args:
        data_dir: Directory containing 'Fraud' and 'Legit' subfolders
        output_dir: Directory to save the analysis results
        
    Returns:
        None
    """
    # Use default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'analysis')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count files in each class
    fraud_files = os.listdir(os.path.join(data_dir, 'Fraud'))
    legit_files = os.listdir(os.path.join(data_dir, 'Legit'))
    
    # Calculate class distribution
    total_files = len(fraud_files) + len(legit_files)
    fraud_ratio = len(fraud_files) / total_files
    legit_ratio = len(legit_files) / total_files
    
    # Log dataset statistics
    logger.info(f"Dataset statistics for {data_dir}:")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Fraud files: {len(fraud_files)} ({fraud_ratio:.2%})")
    logger.info(f"Legit files: {len(legit_files)} ({legit_ratio:.2%})")
    
    # Create class distribution plot
    plt.figure(figsize=(8, 6))
    plt.bar(['Legit', 'Fraud'], [len(legit_files), len(fraud_files)])
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # Save statistics to CSV
    stats_df = pd.DataFrame({
        'Class': ['Legit', 'Fraud', 'Total'],
        'Count': [len(legit_files), len(fraud_files), total_files],
        'Percentage': [legit_ratio * 100, fraud_ratio * 100, 100]
    })
    
    stats_df.to_csv(os.path.join(output_dir, 'dataset_statistics.csv'), index=False)

def balance_dataset(data_dir, output_dir=None, method='oversample', target_ratio=1.0, random_seed=42):
    """
    Balance a dataset by oversampling or undersampling.
    
    Args:
        data_dir: Directory containing 'Fraud' and 'Legit' subfolders
        output_dir: Directory to save the balanced dataset
        method: Balancing method ('oversample', 'undersample', or 'hybrid')
        target_ratio: Target ratio of minority to majority class (1.0 for perfect balance)
        random_seed: Random seed for reproducibility
        
    Returns:
        None
    """
    # Use default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(data_dir), os.path.basename(data_dir) + '_balanced')
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, 'Fraud'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Legit'), exist_ok=True)
    
    # Count files in each class
    fraud_files = os.listdir(os.path.join(data_dir, 'Fraud'))
    legit_files = os.listdir(os.path.join(data_dir, 'Legit'))
    
    # Determine majority and minority classes
    if len(fraud_files) < len(legit_files):
        minority_class = 'Fraud'
        majority_class = 'Legit'
        minority_files = fraud_files
        majority_files = legit_files
    else:
        minority_class = 'Legit'
        majority_class = 'Fraud'
        minority_files = legit_files
        majority_files = fraud_files
    
    # Calculate target counts
    minority_count = len(minority_files)
    majority_count = len(majority_files)
    
    if method == 'oversample':
        # Oversampling: Duplicate minority class samples
        target_minority_count = int(majority_count * target_ratio)
        target_majority_count = majority_count
        
        # Copy all majority class samples
        for file in majority_files:
            shutil.copy2(
                os.path.join(data_dir, majority_class, file),
                os.path.join(output_dir, majority_class, file)
            )
        
        # Copy and duplicate minority class samples
        random.seed(random_seed)
        samples_needed = target_minority_count - minority_count
        
        if samples_needed <= 0:
            # Just copy all minority samples if no duplication needed
            for file in minority_files:
                shutil.copy2(
                    os.path.join(data_dir, minority_class, file),
                    os.path.join(output_dir, minority_class, file)
                )
        else:
            # Copy all original minority samples
            for file in minority_files:
                shutil.copy2(
                    os.path.join(data_dir, minority_class, file),
                    os.path.join(output_dir, minority_class, file)
                )
            
            # Duplicate samples until target count is reached
            duplicate_files = random.choices(minority_files, k=samples_needed)
            
            for i, file in enumerate(duplicate_files):
                # Create new filename for duplicate
                filename, extension = os.path.splitext(file)
                new_filename = f"{filename}_dup{i+1}{extension}"
                
                # Copy file with new name
                shutil.copy2(
                    os.path.join(data_dir, minority_class, file),
                    os.path.join(output_dir, minority_class, new_filename)
                )
    
    elif method == 'undersample':
        # Undersampling: Reduce majority class samples
        target_majority_count = int(minority_count / target_ratio)
        target_minority_count = minority_count
        
        # Copy all minority class samples
        for file in minority_files:
            shutil.copy2(
                os.path.join(data_dir, minority_class, file),
                os.path.join(output_dir, minority_class, file)
            )
        
        # Randomly select majority class samples
        random.seed(random_seed)
        selected_majority_files = random.sample(majority_files, target_majority_count)
        
        # Copy selected majority class samples
        for file in selected_majority_files:
            shutil.copy2(
                os.path.join(data_dir, majority_class, file),
                os.path.join(output_dir, majority_class, file)
            )
    
    elif method == 'hybrid':
        # Hybrid approach: Undersample majority and oversample minority
        target_count = int((minority_count + majority_count) / 2)
        
        # Undersample majority class
        random.seed(random_seed)
        selected_majority_files = random.sample(majority_files, target_count)
        
        # Copy selected majority class samples
        for file in selected_majority_files:
            shutil.copy2(
                os.path.join(data_dir, majority_class, file),
                os.path.join(output_dir, majority_class, file)
            )
        
        # Oversample minority class
        samples_needed = target_count - minority_count
        
        # Copy all original minority samples
        for file in minority_files:
            shutil.copy2(
                os.path.join(data_dir, minority_class, file),
                os.path.join(output_dir, minority_class, file)
            )
        
        if samples_needed > 0:
            # Duplicate samples until target count is reached
            duplicate_files = random.choices(minority_files, k=samples_needed)
            
            for i, file in enumerate(duplicate_files):
                # Create new filename for duplicate
                filename, extension = os.path.splitext(file)
                new_filename = f"{filename}_dup{i+1}{extension}"
                
                # Copy file with new name
                shutil.copy2(
                    os.path.join(data_dir, minority_class, file),
                    os.path.join(output_dir, minority_class, new_filename)
                )
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Log dataset statistics
    fraud_output_count = len(os.listdir(os.path.join(output_dir, 'Fraud')))
    legit_output_count = len(os.listdir(os.path.join(output_dir, 'Legit')))
    
    logger.info(f"Balanced dataset created in {output_dir}")
    logger.info(f"Original dataset: {len(fraud_files)} fraud, {len(legit_files)} legit")
    logger.info(f"Balanced dataset: {fraud_output_count} fraud, {legit_output_count} legit")

def augment_data(data_dir, output_dir=None, augmentation_factor=2, random_seed=42):
    """
    Augment data by creating modified copies of existing samples.
    
    Args:
        data_dir: Directory containing 'Fraud' and 'Legit' subfolders
        output_dir: Directory to save the augmented dataset
        augmentation_factor: Number of augmented samples to create per original sample
        random_seed: Random seed for reproducibility
        
    Returns:
        None
    """
    # Use default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(data_dir), os.path.basename(data_dir) + '_augmented')
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, 'Fraud'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Legit'), exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Define augmentation functions for videos
    def augment_video(video_path, output_path):
        """Apply random augmentations to a video."""
        import cv2
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Random augmentation parameters
        flip_horizontal = np.random.random() > 0.5
        brightness_factor = np.random.uniform(0.8, 1.2)
        contrast_factor = np.random.uniform(0.8, 1.2)
        
        # Process video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply augmentations
            if flip_horizontal:
                frame = cv2.flip(frame, 1)
            
            # Adjust brightness and contrast
            frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=(brightness_factor - 1.0) * 127)
            
            # Write frame to output video
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        return True
    
    # Define augmentation functions for audio
    def augment_audio(audio_path, output_path):
        """Apply random augmentations to an audio file."""
        import librosa
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Random augmentation parameters
        time_stretch_factor = np.random.uniform(0.9, 1.1)
        pitch_shift_steps = np.random.uniform(-2, 2)
        
        # Apply time stretching
        y_stretched = librosa.effects.time_stretch(y, rate=time_stretch_factor)
        
        # Apply pitch shifting
        y_shifted = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=pitch_shift_steps)
        
        # Save augmented audio
        sf.write(output_path, y_shifted, sr)
        
        return True
    
    # Augment each class
    for class_name in ['Fraud', 'Legit']:
        class_dir = os.path.join(data_dir, class_name)
        files = os.listdir(class_dir)
        
        # Copy original files to output directory
        for file in files:
            file_path = os.path.join(class_dir, file)
            output_file_path = os.path.join(output_dir, class_name, file)
            
            shutil.copy2(file_path, output_file_path)
        
        # Create augmented versions
        for file in files:
            file_path = os.path.join(class_dir, file)
            
            for i in range(augmentation_factor):
                # Create new filename for augmented sample
                filename, extension = os.path.splitext(file)
                new_filename = f"{filename}_aug{i+1}{extension}"
                output_file_path = os.path.join(output_dir, class_name, new_filename)
                
                # Determine file type and apply appropriate augmentation
                if extension.lower() in ['.mp4', '.avi', '.mov']:
                    augment_video(file_path, output_file_path)
                elif extension.lower() in ['.wav', '.mp3', '.ogg']:
                    augment_audio(file_path, output_file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
    
    # Log dataset statistics
    fraud_output_count = len(os.listdir(os.path.join(output_dir, 'Fraud')))
    legit_output_count = len(os.listdir(os.path.join(output_dir, 'Legit')))
    
    logger.info(f"Augmented dataset created in {output_dir}")
    logger.info(f"Original dataset: {len(os.listdir(os.path.join(data_dir, 'Fraud')))} fraud, {len(os.listdir(os.path.join(data_dir, 'Legit')))} legit")
    logger.info(f"Augmented dataset: {fraud_output_count} fraud, {legit_output_count} legit")

def create_tfrecord(data_dir, output_path, preprocessor=None):
    """
    Create a TFRecord file from a dataset.
    
    Args:
        data_dir: Directory containing 'Fraud' and 'Legit' subfolders
        output_path: Path to save the TFRecord file
        preprocessor: Function to preprocess data before writing to TFRecord
        
    Returns:
        None
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(output_path) as writer:
        # Process each class
        for class_name in ['Fraud', 'Legit']:
            class_dir = os.path.join(data_dir, class_name)
            files = os.listdir(class_dir)
            
            for file in files:
                file_path = os.path.join(class_dir, file)
                
                # Apply preprocessing if provided
                if preprocessor is not None:
                    data = preprocessor(file_path)
                else:
                    # Read file as binary data
                    with open(file_path, 'rb') as f:
                        data = f.read()
                
                # Create label
                label = 1 if class_name == 'Fraud' else 0
                
                # Create feature dictionary
                feature = {
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file.encode()])),
                }
                
                # Create example
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                # Write example to TFRecord file
                writer.write(example.SerializeToString())
    
    logger.info(f"TFRecord file created: {output_path}")

def synchronize_video_audio_datasets(video_dir, audio_dir, output_dir=None):
    """
    Synchronize video and audio datasets for multimodal training.
    
    Args:
        video_dir: Directory containing video data with 'Fraud' and 'Legit' subfolders
        audio_dir: Directory containing audio data with 'Fraud' and 'Legit' subfolders
        output_dir: Directory to save the synchronized dataset information
        
    Returns:
        synchronized_data: List of (video_path, audio_path, label) tuples
    """
    # Use default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_dir), 'synchronized_dataset')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video and audio files
    video_fraud_files = [os.path.join(video_dir, 'Fraud', f) for f in os.listdir(os.path.join(video_dir, 'Fraud'))]
    video_legit_files = [os.path.join(video_dir, 'Legit', f) for f in os.listdir(os.path.join(video_dir, 'Legit'))]
    
    audio_fraud_files = [os.path.join(audio_dir, 'Fraud', f) for f in os.listdir(os.path.join(audio_dir, 'Fraud'))]
    audio_legit_files = [os.path.join(audio_dir, 'Legit', f) for f in os.listdir(os.path.join(audio_dir, 'Legit'))]
    
    # Check if datasets have the same number of files
    if len(video_fraud_files) != len(audio_fraud_files) or len(video_legit_files) != len(audio_legit_files):
        logger.warning("Video and audio datasets have different numbers of files. Using minimum count for each class.")
    
    # Create synchronized dataset
    synchronized_data = []
    
    # Synchronize fraud data
    min_fraud_count = min(len(video_fraud_files), len(audio_fraud_files))
    for i in range(min_fraud_count):
        synchronized_data.append((video_fraud_files[i], audio_fraud_files[i], 1))
    
    # Synchronize legit data
    min_legit_count = min(len(video_legit_files), len(audio_legit_files))
    for i in range(min_legit_count):
        synchronized_data.append((video_legit_files[i], audio_legit_files[i], 0))
    
    # Save synchronized dataset information to CSV
    df = pd.DataFrame(synchronized_data, columns=['video_path', 'audio_path', 'label'])
    df.to_csv(os.path.join(output_dir, 'synchronized_dataset.csv'), index=False)
    
    logger.info(f"Synchronized dataset created with {min_fraud_count} fraud and {min_legit_count} legit samples")
    logger.info(f"Dataset information saved to {os.path.join(output_dir, 'synchronized_dataset.csv')}")
    
    return synchronized_data

def create_synchronized_tfrecord(synchronized_data, output_path, video_preprocessor=None, audio_preprocessor=None):
    """
    Create a TFRecord file from synchronized video and audio data.
    
    Args:
        synchronized_data: List of (video_path, audio_path, label) tuples
        output_path: Path to save the TFRecord file
        video_preprocessor: Function to preprocess video data
        audio_preprocessor: Function to preprocess audio data
        
    Returns:
        None
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create TFRecord writer
    with tf.io.TFRecordWriter(output_path) as writer:
        for video_path, audio_path, label in synchronized_data:
            # Process video data
            if video_preprocessor is not None:
                video_data = video_preprocessor(video_path)
            else:
                # Read file as binary data
                with open(video_path, 'rb') as f:
                    video_data = f.read()
            
            # Process audio data
            if audio_preprocessor is not None:
                audio_data = audio_preprocessor(audio_path)
            else:
                # Read file as binary data
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
            
            # Create feature dictionary
            feature = {
                'video_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_data])),
                'audio_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_data])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'video_filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.basename(video_path).encode()])),
                'audio_filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[os.path.basename(audio_path).encode()])),
            }
            
            # Create example
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Write example to TFRecord file
            writer.write(example.SerializeToString())
    
    logger.info(f"Synchronized TFRecord file created: {output_path}")