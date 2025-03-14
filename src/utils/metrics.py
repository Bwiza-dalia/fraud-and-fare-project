import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate metrics for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Threshold for classification
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Store metrics in dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names=None, output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        output_path: Path to save the plot
        
    Returns:
        None
    """
    if class_names is None:
        class_names = ['Legit', 'Fraud']
    
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
    
    # Save figure if output_path is provided
    if output_path is not None:
        plt.savefig(output_path)
        logger.info(f"Confusion matrix saved to {output_path}")
    
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, output_path=None):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under the ROC curve
        output_path: Path to save the plot
        
    Returns:
        None
    """
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Add labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Save figure if output_path is provided
    if output_path is not None:
        plt.savefig(output_path)
        logger.info(f"ROC curve saved to {output_path}")
    
    plt.close()

def plot_precision_recall_curve(y_true, y_pred, output_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        output_path: Path to save the plot
        
    Returns:
        None
    """
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot precision-recall curve
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    
    # Add labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    
    # Save figure if output_path is provided
    if output_path is not None:
        plt.savefig(output_path)
        logger.info(f"Precision-recall curve saved to {output_path}")
    
    plt.close()

def find_optimal_threshold(y_true, y_pred):
    """
    Find the optimal threshold for classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        optimal_threshold: Optimal threshold
    """
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find the threshold that maximizes F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    
    return optimal_threshold

def evaluate_model_on_dataset(model, dataset, output_dir=None):
    """
    Evaluate a model on a dataset and generate evaluation metrics and plots.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate
        output_dir: Directory to save evaluation results
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Use default output directory if not provided
    if output_dir is None:
        output_dir = 'evaluation_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect true labels and predictions
    y_true = []
    y_pred = []
    
    for x, y in dataset:
        # Get predictions
        batch_pred = model.predict(x)
        
        # Append to lists
        y_true.extend(y.numpy())
        y_pred.extend(batch_pred[:, 1])  # Assuming binary classification and second column is positive class
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_true, y_pred)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, threshold=optimal_threshold)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        output_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        metrics['fpr'], metrics['tpr'], metrics['auc'],
        output_path=os.path.join(output_dir, 'roc_curve.png')
    )
    
    # Plot precision-recall curve
    plot_precision_recall_curve(
        y_true, y_pred,
        output_path=os.path.join(output_dir, 'precision_recall_curve.png')
    )
    
    # Print classification report
    report = classification_report(y_true, (np.array(y_pred) > optimal_threshold).astype(int))
    logger.info(f"Classification report:\n{report}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Optimal Threshold'],
        'Value': [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['auc'],
            optimal_threshold
        ]
    })
    
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    logger.info(f"Metrics saved to {os.path.join(output_dir, 'metrics.csv')}")
    
    return metrics

def evaluate_ensemble_on_dataset(video_model, audio_model, dataset, weights=(0.6, 0.4), output_dir=None):
    """
    Evaluate an ensemble model on a dataset and generate evaluation metrics and plots.
    
    Args:
        video_model: Trained video model
        audio_model: Trained audio model
        dataset: Tuple of (video_dataset, audio_dataset) to evaluate
        weights: Tuple of (video_weight, audio_weight) for ensemble
        output_dir: Directory to save evaluation results
        
    Returns:
        metrics: Dictionary of metrics
    """
    # Use default output directory if not provided
    if output_dir is None:
        output_dir = 'ensemble_evaluation_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack datasets
    video_dataset, audio_dataset = dataset
    
    # Collect true labels and predictions
    y_true = []
    video_pred = []
    audio_pred = []
    
    # Get video predictions
    for x, y in video_dataset:
        # Get predictions
        batch_pred = video_model.predict(x)
        
        # Append to lists
        y_true.extend(y.numpy())
        video_pred.extend(batch_pred[:, 1])  # Assuming binary classification and second column is positive class
    
    # Get audio predictions
    audio_true = []
    for x, y in audio_dataset:
        # Get predictions
        batch_pred = audio_model.predict(x)
        
        # Append to lists
        audio_true.extend(y.numpy())
        audio_pred.extend(batch_pred[:, 1])
    
    # Ensure labels match
    assert y_true == audio_true, "Video and audio datasets have different labels"
    
    # Apply weights to create ensemble predictions
    video_weight, audio_weight = weights
    ensemble_pred = np.array(video_pred) * video_weight + np.array(audio_pred) * audio_weight
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_true, ensemble_pred)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, ensemble_pred, threshold=optimal_threshold)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        output_path=os.path.join(output_dir, 'ensemble_confusion_matrix.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        metrics['fpr'], metrics['tpr'], metrics['auc'],
        output_path=os.path.join(output_dir, 'ensemble_roc_curve.png')
    )
    
    # Plot precision-recall curve
    plot_precision_recall_curve(
        y_true, ensemble_pred,
        output_path=os.path.join(output_dir, 'ensemble_precision_recall_curve.png')
    )
    
    # Print classification report
    report = classification_report(y_true, (ensemble_pred > optimal_threshold).astype(int))
    logger.info(f"Ensemble classification report:\n{report}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Optimal Threshold'],
        'Value': [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['auc'],
            optimal_threshold
        ]
    })
    
    metrics_df.to_csv(os.path.join(output_dir, 'ensemble_metrics.csv'), index=False)
    logger.info(f"Ensemble metrics saved to {os.path.join(output_dir, 'ensemble_metrics.csv')}")
    
    # Compare with individual models
    video_metrics = calculate_metrics(y_true, video_pred)
    audio_metrics = calculate_metrics(y_true, audio_pred)
    
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
        'Video Model': [
            video_metrics['accuracy'],
            video_metrics['precision'],
            video_metrics['recall'],
            video_metrics['f1_score'],
            video_metrics['auc']
        ],
        'Audio Model': [
            audio_metrics['accuracy'],
            audio_metrics['precision'],
            audio_metrics['recall'],
            audio_metrics['f1_score'],
            audio_metrics['auc']
        ],
        'Ensemble Model': [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['auc']
        ]
    })
    
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    logger.info(f"Model comparison saved to {os.path.join(output_dir, 'model_comparison.csv')}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Convert to long format for seaborn
    comparison_long = pd.melt(
        comparison_df,
        id_vars=['Metric'],
        value_vars=['Video Model', 'Audio Model', 'Ensemble Model'],
        var_name='Model',
        value_name='Score'
    )
    
    # Plot comparison
    sns.barplot(x='Metric', y='Score', hue='Model', data=comparison_long)
    
    # Add labels
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    logger.info(f"Model comparison plot saved to {os.path.join(output_dir, 'model_comparison.png')}")
    plt.close()
    
    return metrics