#!/usr/bin/env python
# Main script for fraud detection system

import argparse
import os
import logging
import tensorflow as tf
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/fraud_detection_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
from src.training.train_video import train_video_model
from src.training.train_audio import train_audio_model
from src.training.train_fusion import train_fusion_model
from src.utils.visualization import visualize_results
from deploy.inference import run_inference
import config

def set_gpu_memory_growth():
    """Configure GPU to allocate memory as needed."""
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger.info(f"Found {len(physical_devices)} GPU(s). Memory growth enabled.")
    except:
        logger.warning("Failed to set memory growth. Using default GPU configuration.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fraud Detection System for Public Transport')
    
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'test', 'deploy'],
                      help='Operation mode: train, test or deploy')
    
    parser.add_argument('--model', type=str, default='all',
                      choices=['video', 'audio', 'fusion', 'all'],
                      help='Model to train/test/deploy')
    
    parser.add_argument('--epochs', type=int, default=None,
                      help='Override number of epochs in config')
    
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Override batch size in config')
    
    parser.add_argument('--lr', type=float, default=None,
                      help='Override learning rate in config')
    
    parser.add_argument('--video_path', type=str, default=None,
                      help='Path to video file for inference')
    
    parser.add_argument('--audio_path', type=str, default=None,
                      help='Path to audio file for inference')
    
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize results')
                      
    return parser.parse_args()

def override_config(args):
    """Override config parameters with command line arguments."""
    if args.epochs:
        config.VIDEO_CONFIG['epochs'] = args.epochs
        config.AUDIO_CONFIG['epochs'] = args.epochs
        config.FUSION_CONFIG['epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    
    if args.batch_size:
        config.VIDEO_CONFIG['batch_size'] = args.batch_size
        config.AUDIO_CONFIG['batch_size'] = args.batch_size
        config.FUSION_CONFIG['batch_size'] = args.batch_size
        logger.info(f"Overriding batch_size: {args.batch_size}")
    
    if args.lr:
        config.VIDEO_CONFIG['learning_rate'] = args.lr
        config.AUDIO_CONFIG['learning_rate'] = args.lr
        config.FUSION_CONFIG['learning_rate'] = args.lr
        logger.info(f"Overriding learning_rate: {args.lr}")

def main():
    """Main function to run the fraud detection system."""
    args = parse_arguments()
    override_config(args)
    set_gpu_memory_growth()
    
    logger.info(f"Running in {args.mode} mode for {args.model} model")
    
    # Training mode
    if args.mode == 'train':
        if args.model in ['video', 'all']:
            logger.info("Training video model...")
            video_model, video_history = train_video_model()
            
        if args.model in ['audio', 'all']:
            logger.info("Training audio model...")
            audio_model, audio_history = train_audio_model()
            
        if args.model in ['fusion', 'all']:
            logger.info("Training fusion model...")
            fusion_model, fusion_history = train_fusion_model()
        
        if args.visualize:
            logger.info("Visualizing training results...")
            if args.model in ['video', 'all']:
                visualize_results(video_history, model_type='video')
            if args.model in ['audio', 'all']:
                visualize_results(audio_history, model_type='audio')
            if args.model in ['fusion', 'all']:
                visualize_results(fusion_history, model_type='fusion')
    
    # Test mode
    elif args.mode == 'test':
        # Import here to avoid loading test modules during training
        from src.training.train_video import evaluate_video_model
        from src.training.train_audio import evaluate_audio_model
        from src.training.train_fusion import evaluate_fusion_model
        
        if args.model in ['video', 'all']:
            logger.info("Evaluating video model...")
            video_results = evaluate_video_model()
            logger.info(f"Video model results: {video_results}")
            
        if args.model in ['audio', 'all']:
            logger.info("Evaluating audio model...")
            audio_results = evaluate_audio_model()
            logger.info(f"Audio model results: {audio_results}")
            
        if args.model in ['fusion', 'all']:
            logger.info("Evaluating fusion model...")
            fusion_results = evaluate_fusion_model()
            logger.info(f"Fusion model results: {fusion_results}")
    
    # Deploy mode
    elif args.mode == 'deploy':
        if not args.video_path and not args.audio_path:
            logger.error("No input file specified. Please provide --video_path or --audio_path")
            return
        
        logger.info("Running inference...")
        results = run_inference(
            video_path=args.video_path,
            audio_path=args.audio_path,
            model_type=args.model
        )
        
        logger.info(f"Inference results: {results}")
        
        if args.visualize and args.video_path:
            from src.utils.visualization import visualize_inference
            logger.info("Visualizing inference results...")
            visualize_inference(args.video_path, results)

if __name__ == "__main__":
    main()