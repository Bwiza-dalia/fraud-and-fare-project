import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
import logging
from tqdm import tqdm
import random
import shutil

# Configure logging
logger = logging.getLogger(__name__)

def extract_frames(video_path, target_frames=64, target_size=(224, 224)):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        target_frames: Number of frames to extract
        target_size: Target frame size (height, width)
        
    Returns:
        frames: List of extracted frames
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            logger.warning(f"Video {video_path} has 0 frames")
            return None
        
        # Calculate indices of frames to extract
        if frame_count <= target_frames:
            # If video has fewer frames than target, duplicate frames
            indices = list(range(frame_count))
            # Duplicate last frames to reach target_frames
            indices.extend([frame_count-1] * (target_frames - frame_count))
        else:
            # Sample frames uniformly
            indices = np.linspace(0, frame_count-1, target_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(frame, target_size)
                frames.append(frame)
            else:
                # If frame reading fails, append a blank frame
                logger.warning(f"Failed to read frame {idx} from {video_path}")
                frames.append(np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8))
        
        cap.release()
        
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {str(e)}")
        return None
    
    return np.array(frames)

def create_video_dataset(video_dir, target_frames=64, target_size=(224, 224), batch_size=8, 
                        val_split=0.2, test_split=0.1, random_state=42, balance_classes=True):
    """
    Create a dataset from videos for fraud detection.
    
    Args:
        video_dir: Directory containing 'Fraud' and 'Legit' subfolders with videos
        target_frames: Number of frames to extract from each video
        target_size: Target frame size (height, width)
        batch_size: Batch size for training
        val_split: Validation split ratio
        test_split: Test split ratio
        random_state: Random seed for reproducibility
        balance_classes: Whether to balance classes
        
    Returns:
        train_dataset, val_dataset, test_dataset: TensorFlow datasets for training, validation, and testing
    """
    # Create lists to store video paths and labels
    video_paths = []
    labels = []
    
    # Get paths for fraud videos
    fraud_dir = os.path.join(video_dir, 'Fraud')
    for video_file in os.listdir(fraud_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(fraud_dir, video_file))
            labels.append(1)  # 1 for fraud
    
    # Get paths for legitimate videos
    legit_dir = os.path.join(video_dir, 'Legit')
    for video_file in os.listdir(legit_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(legit_dir, video_file))
            labels.append(0)  # 0 for legitimate
    
    # Balance classes if needed
    if balance_classes:
        video_paths, labels = balance_dataset(video_paths, labels)
    
    # Split data into train, validation, and test sets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels, test_size=(val_split + test_split), 
        random_state=random_state, stratify=labels
    )
    
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1-val_ratio),
        random_state=random_state, stratify=temp_labels
    )
    
    logger.info(f"Dataset split: {len(train_paths)} train, {len(val_paths)} validation, {len(test_paths)} test")
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    
    # Preprocess datasets
    train_dataset = train_dataset.map(
        lambda x, y: tf.py_function(preprocess_video, [x, y, target_frames, target_size, True], 
                                    [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    val_dataset = val_dataset.map(
        lambda x, y: tf.py_function(preprocess_video, [x, y, target_frames, target_size, False], 
                                    [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    test_dataset = test_dataset.map(
        lambda x, y: tf.py_function(preprocess_video, [x, y, target_frames, target_size, False], 
                                    [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Cache, batch, and prefetch datasets
    train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def preprocess_video(video_path, label, target_frames, target_size, augment):
    """
    Preprocess a video for the model.
    
    Args:
        video_path: Path to video file
        label: Video label
        target_frames: Number of frames to extract
        target_size: Target frame size (height, width)
        augment: Whether to apply data augmentation
        
    Returns:
        frames: Preprocessed video frames
        label: Video label
    """
    try:
        # Extract frames
        frames = extract_frames(
            video_path.numpy().decode(), 
            target_frames=target_frames, 
            target_size=target_size
        )
        
        if frames is None:
            # If frame extraction failed, return a blank video
            frames = np.zeros((target_frames, target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # Apply augmentation if needed
        if augment:
            frames = augment_video(frames)
        
        # Preprocess frames (normalize pixel values)
        frames = preprocess_input(frames)
        
        return frames, label
    
    except Exception as e:
        logger.error(f"Error preprocessing video: {str(e)}")
        # Return a blank video in case of error
        return np.zeros((target_frames, target_size[0], target_size[1], 3), dtype=np.float32), label

def augment_video(frames):
    """
    Apply data augmentation to video frames.
    
    Args:
        frames: Video frames to augment
        
    Returns:
        frames: Augmented video frames
    """
    # Random horizontal flip
    if random.random() > 0.5:
        frames = frames[:, :, ::-1, :]
    
    # Random brightness
    if random.random() > 0.5:
        delta = 0.2 * random.random() - 0.1
        frames = np.clip(frames + delta, 0, 255)
    
    # Random contrast
    if random.random() > 0.5:
        factor = 1.0 + 0.2 * random.random() - 0.1
        frames = np.clip(factor * (frames - 128) + 128, 0, 255)
    
    # Random temporal shift
    if random.random() > 0.5:
        shift = random.randint(1, 10)
        if random.random() > 0.5:
            # Shift right
            frames = np.concatenate([frames[-shift:], frames[:-shift]], axis=0)
        else:
            # Shift left
            frames = np.concatenate([frames[shift:], frames[:shift]], axis=0)
    
    # Random temporal crop and resize
    if random.random() > 0.5:
        n_frames = frames.shape[0]
        start = random.randint(0, n_frames // 4)
        end = random.randint(3 * n_frames // 4, n_frames)
        frames = frames[start:end]
        # Resize back to original number of frames
        indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        frames = frames[indices]
    
    return frames

def balance_dataset(video_paths, labels):
    """
    Balance the dataset by oversampling the minority class.
    
    Args:
        video_paths: List of video paths
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
    balanced_paths = [video_paths[i] for i in all_indices]
    balanced_labels = [labels[i] for i in all_indices]
    
    logger.info(f"Balanced dataset: {len(balanced_paths)} videos, {sum(balanced_labels)} fraud, {len(balanced_labels) - sum(balanced_labels)} legitimate")
    
    return balanced_paths, balanced_labels

def prepare_video_directory(src_dir, dst_dir, target_frames=16):
    """
    Prepare video directory by extracting frames from videos and saving them.
    This is useful for models that require frame input rather than video input.
    
    Args:
        src_dir: Source directory containing 'Fraud' and 'Legit' subfolders with videos
        dst_dir: Destination directory to save extracted frames
        target_frames: Number of frames to extract from each video
    """
    # Create destination directory
    os.makedirs(dst_dir, exist_ok=True)
    
    # Create 'Fraud' and 'Legit' subfolders
    fraud_dst_dir = os.path.join(dst_dir, 'Fraud')
    legit_dst_dir = os.path.join(dst_dir, 'Legit')
    os.makedirs(fraud_dst_dir, exist_ok=True)
    os.makedirs(legit_dst_dir, exist_ok=True)
    
    # Process fraud videos
    fraud_src_dir = os.path.join(src_dir, 'Fraud')
    for video_file in tqdm(os.listdir(fraud_src_dir), desc="Processing fraud videos"):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(fraud_src_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_dst_dir = os.path.join(fraud_dst_dir, video_name)
            os.makedirs(video_dst_dir, exist_ok=True)
            
            # Extract frames
            frames = extract_frames(video_path, target_frames=target_frames)
            if frames is not None:
                # Save frames
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(video_dst_dir, f"frame_{i:04d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Process legitimate videos
    legit_src_dir = os.path.join(src_dir, 'Legit')
    for video_file in tqdm(os.listdir(legit_src_dir), desc="Processing legitimate videos"):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(legit_src_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_dst_dir = os.path.join(legit_dst_dir, video_name)
            os.makedirs(video_dst_dir, exist_ok=True)
            
            # Extract frames
            frames = extract_frames(video_path, target_frames=target_frames)
            if frames is not None:
                # Save frames
                for i, frame in enumerate(frames):
                    frame_path = os.path.join(video_dst_dir, f"frame_{i:04d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Video frames extracted and saved to {dst_dir}")

def create_tfrecord_dataset(video_dir, output_dir, target_frames=64, target_size=(224, 224)):
    """
    Create TFRecord dataset from videos for faster loading.
    
    Args:
        video_dir: Directory containing 'Fraud' and 'Legit' subfolders with videos
        output_dir: Directory to save TFRecord files
        target_frames: Number of frames to extract from each video
        target_size: Target frame size (height, width)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video paths and labels
    video_paths = []
    labels = []
    
    # Get paths for fraud videos
    fraud_dir = os.path.join(video_dir, 'Fraud')
    for video_file in os.listdir(fraud_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(fraud_dir, video_file))
            labels.append(1)  # 1 for fraud
    
    # Get paths for legitimate videos
    legit_dir = os.path.join(video_dir, 'Legit')
    for video_file in os.listdir(legit_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_paths.append(os.path.join(legit_dir, video_file))
            labels.append(0)  # 0 for legitimate
    
    # Split data into train, validation, and test sets
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Function to create TFRecord file
    def create_tfrecord(video_paths, labels, output_path):
        with tf.io.TFRecordWriter(output_path) as writer:
            for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths)):
                # Extract frames
                frames = extract_frames(video_path, target_frames, target_size)
                if frames is None:
                    continue
                
                # Create feature
                feature = {
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'frames': tf.train.Feature(bytes_list=tf.train.BytesList(value=[frames.tobytes()])),
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=list(frames.shape))),
                }
                
                # Create example
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                # Write example
                writer.write(example.SerializeToString())
    
    # Create TFRecord files
    train_output_path = os.path.join(output_dir, 'train.tfrecord')
    val_output_path = os.path.join(output_dir, 'val.tfrecord')
    test_output_path = os.path.join(output_dir, 'test.tfrecord')
    
    logger.info("Creating train TFRecord file...")
    create_tfrecord(train_paths, train_labels, train_output_path)
    
    logger.info("Creating validation TFRecord file...")
    create_tfrecord(val_paths, val_labels, val_output_path)
    
    logger.info("Creating test TFRecord file...")
    create_tfrecord(test_paths, test_labels, test_output_path)
    
    logger.info(f"TFRecord files created and saved to {output_dir}")