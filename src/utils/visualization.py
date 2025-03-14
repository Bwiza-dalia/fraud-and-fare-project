import matplotlib.pyplot as plt
import numpy as np
import cv2
import librosa
import librosa.display
import os
import logging
import pandas as pd
import seaborn as sns
from tensorflow.keras.utils import plot_model

# Configure logging
logger = logging.getLogger(__name__)

def visualize_results(history, model_type='video', output_dir=None):
    """
    Visualize training results.
    
    Args:
        history: Training history
        model_type: Type of model ('video', 'audio', 'fusion')
        output_dir: Directory to save visualization results
        
    Returns:
        None
    """
    # Use default output directory if not provided
    if output_dir is None:
        output_dir = f'{model_type}_visualization'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type.capitalize()} Model Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_type}_loss.png'))
    plt.close()
    
    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_type.capitalize()} Model Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_type}_accuracy.png'))
    plt.close()
    
    # Check if precision and recall are in history
    if 'precision' in history.history and 'recall' in history.history:
        # Plot training and validation precision
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['precision'], label='Training Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.title(f'{model_type.capitalize()} Model Training and Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_type}_precision.png'))
        plt.close()
        
        # Plot training and validation recall
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['recall'], label='Training Recall')
        plt.plot(history.history['val_recall'], label='Validation Recall')
        plt.title(f'{model_type.capitalize()} Model Training and Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_type}_recall.png'))
        plt.close()
    
    # Check if AUC is in history
    if 'auc' in history.history:
        # Plot training and validation AUC
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['auc'], label='Training AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title(f'{model_type.capitalize()} Model Training and Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_type}_auc.png'))
        plt.close()
    
    # Check if F1 score is in history
    if 'f1_score' in history.history:
        # Plot training and validation F1 score
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['f1_score'], label='Training F1 Score')
        plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
        plt.title(f'{model_type.capitalize()} Model Training and Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{model_type}_f1_score.png'))
        plt.close()
    
    logger.info(f"Visualization results saved to {output_dir}")

def visualize_model_architecture(model, output_path=None):
    """
    Visualize model architecture.
    
    Args:
        model: Keras model
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Use default output path if not provided
    if output_path is None:
        output_path = 'model_architecture.png'
    
    # Plot model architecture
    plot_model(
        model,
        to_file=output_path,
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_dtype=True
    )
    
    logger.info(f"Model architecture visualization saved to {output_path}")

def visualize_video_frames(video_path, output_dir=None, num_frames=16, frame_interval=None):
    """
    Visualize frames from a video.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the frame images
        num_frames: Number of frames to extract and visualize
        frame_interval: Interval between frames (if None, calculate based on video length)
        
    Returns:
        None
    """
    # Use default output directory if not provided
    if output_dir is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f'{video_name}_frames'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video: {video_path}")
        return None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval if not provided
    if frame_interval is None:
        frame_interval = max(1, frame_count // num_frames)
    
    # Extract and save frames
    frames = []
    for i in range(num_frames):
        # Set frame position
        frame_position = min(i * frame_interval, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{i:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        frames.append(frame)
    
    # Release video
    cap.release()
    
    # Create grid of frames
    rows = int(np.ceil(np.sqrt(len(frames))))
    cols = int(np.ceil(len(frames) / rows))
    
    plt.figure(figsize=(cols * 4, rows * 3))
    
    for i, frame in enumerate(frames):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(f'Frame {i * frame_interval}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frame_grid.png'))
    plt.close()
    
    logger.info(f"Video frames saved to {output_dir}")

def visualize_audio_spectrogram(audio_path, output_dir=None, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """
    Visualize audio spectrogram.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save the visualization
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        None
    """
    # Use default output directory if not provided
    if output_dir is None:
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = f'{audio_name}_spectrogram'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Create figure with subplots
    plt.figure(figsize=(12, 8))
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Convert to dB scale
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Plot mel spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        mel_spectrogram_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Bands')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectrogram.png'))
    plt.close()
    
    # Plot additional features
    plt.figure(figsize=(12, 10))
    
    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Plot MFCC
    plt.subplot(3, 1, 1)
    librosa.display.specshow(
        mfcc,
        sr=sr,
        x_axis='time',
        cmap='coolwarm'
    )
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    
    # Compute chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Plot chroma
    plt.subplot(3, 1, 2)
    librosa.display.specshow(
        chroma,
        sr=sr,
        x_axis='time',
        y_axis='chroma',
        cmap='coolwarm'
    )
    plt.colorbar()
    plt.title('Chroma')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch Class')
    
    # Compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Plot spectral contrast
    plt.subplot(3, 1, 3)
    librosa.display.specshow(
        contrast,
        sr=sr,
        x_axis='time',
        cmap='coolwarm'
    )
    plt.colorbar()
    plt.title('Spectral Contrast')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency Bands')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'audio_features.png'))
    plt.close()
    
    logger.info(f"Audio visualization saved to {output_dir}")

def visualize_confusion_matrix(cm, class_names=['Legit', 'Fraud'], output_path=None):
    """
    Visualize confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Use default output path if not provided
    if output_path is None:
        output_path = 'confusion_matrix.png'
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    
    # Add labels
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Confusion matrix visualization saved to {output_path}")

def visualize_inference(video_path, results, output_path=None):
    """
    Visualize model inference on a video.
    
    Args:
        video_path: Path to the video file
        results: Inference results (dictionary with predictions)
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Use default output path if not provided
    if output_path is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f'{video_name}_inference.mp4'
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video: {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get prediction
    pred_class = np.argmax(results)
    confidence = results[pred_class]
    
    # Process video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add prediction text
        text = f"Prediction: {'Fraud' if pred_class == 1 else 'Legit'} ({confidence:.2f})"
        cv2.putText(
            frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255) if pred_class == 1 else (0, 255, 0), 2
        )
        
        # Write frame to output video
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    logger.info(f"Inference visualization saved to {output_path}")

def plot_model_comparison(metrics_dict, output_path=None):
    """
    Plot comparison of different models.
    
    Args:
        metrics_dict: Dictionary of model metrics
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Use default output path if not provided
    if output_path is None:
        output_path = 'model_comparison.png'
    
    # Extract metrics for each model
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    # Create dataframe for plotting
    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric.capitalize(),
                'Value': metrics_dict[model][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot comparison
    sns.barplot(x='Metric', y='Value', hue='Model', data=df)
    
    # Add labels
    plt.title('Model Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(title='Model')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Model comparison visualization saved to {output_path}")

def visualize_attention_maps(model, video_frames, audio_spectrogram, output_path=None):
    """
    Visualize attention maps for multimodal fusion model.
    
    Args:
        model: Trained multimodal fusion model
        video_frames: Video frames (batch_size, frames, height, width, channels)
        audio_spectrogram: Audio spectrogram (batch_size, height, width, channels)
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Use default output path if not provided
    if output_path is None:
        output_path = 'attention_maps.png'
    
    # Create attention model
    attention_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.MultiHeadAttention)]
    )
    
    # Get attention outputs
    attention_outputs = attention_model.predict([video_frames, audio_spectrogram])
    
    # Create figure
    num_attention_layers = len(attention_outputs)
    plt.figure(figsize=(15, 5 * num_attention_layers))
    
    # Plot attention maps
    for i, attention_output in enumerate(attention_outputs):
        # Get attention weights (batch_size, sequence_length, sequence_length)
        attention_weights = attention_output[0]  # First item in attention output is attention weights
        
        # Plot attention map
        plt.subplot(num_attention_layers, 1, i + 1)
        plt.imshow(attention_weights, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Map - Layer {i + 1}')
        plt.xlabel('Sequence Position')
        plt.ylabel('Sequence Position')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Attention maps visualization saved to {output_path}")

def visualize_feature_importance(model, test_data, class_names=['Legit', 'Fraud'], output_path=None):
    """
    Visualize feature importance for fraud detection model.
    
    Args:
        model: Trained model
        test_data: Test data
        class_names: Names of classes
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Use default output path if not provided
    if output_path is None:
        output_path = 'feature_importance.png'
    
    # Use SHAP for feature importance
    try:
        import shap
        
        # Create explainer
        explainer = shap.DeepExplainer(model, test_data)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(test_data)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot feature importance
        shap.summary_plot(shap_values, test_data, class_names=class_names, show=False)
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Feature importance visualization saved to {output_path}")
    
    except ImportError:
        logger.warning("SHAP library not installed. Install with 'pip install shap' to visualize feature importance.")