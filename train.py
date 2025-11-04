"""
Unified training script that routes to the appropriate model training script
based on model type. This script is called by Flask when a user submits a training job.
"""
import os
import sys
import argparse
import json
import shutil
from pathlib import Path

# Note: Paths will be added dynamically in each training function


def train_yolo(job_id, epochs, batch_size, dataset_path, output_dir):
    """
    Train YOLO model for object identification.
    
    Args:
        job_id: Unique job identifier
        epochs: Number of training epochs
        batch_size: Batch size for training
        dataset_path: Path to the extracted dataset
        output_dir: Base directory for outputs
    """
    print(f"[YOLO] Starting training for job {job_id}")
    
    # Add the ObjectIdentification_AI directory to path for imports
    yolo_dir = os.path.join(os.path.dirname(__file__), 'AI models', 'ObjectIdentification_AI')
    if yolo_dir not in sys.path:
        sys.path.insert(0, yolo_dir)
    
    # Import YOLO training function
    from run_yolo11 import train_yolo_model
    
    # Set up output directory
    results_dir = os.path.join(output_dir, job_id, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Change to YOLO directory for relative paths to work
    original_dir = os.getcwd()
    try:
        os.chdir(yolo_dir)
        
        # Train the model
        train_yolo_model(
            dataset_path=os.path.abspath(dataset_path),
            epochs=epochs,
            batch_size=batch_size,
            output_dir=os.path.abspath(results_dir)
        )
    finally:
        os.chdir(original_dir)
    
    print(f"[YOLO] Training completed. Results saved to {results_dir}")
    sys.stdout.flush()  # Ensure output is flushed


def train_resnet(job_id, epochs, batch_size, dataset_path, output_dir):
    """
    Train ResNet model for orientation classification.
    
    Args:
        job_id: Unique job identifier
        epochs: Number of training epochs
        batch_size: Batch size for training
        dataset_path: Path to the extracted dataset
        output_dir: Base directory for outputs
    """
    print(f"[ResNet] Starting training for job {job_id}")
    
    # Add the Orientation_AI directory to path for imports
    orientation_dir = os.path.join(os.path.dirname(__file__), 'AI models', 'Orientation_AI')
    if orientation_dir not in sys.path:
        sys.path.insert(0, orientation_dir)
    
    # Import training function
    from train_classification import train_orientation_model
    
    # Set up output directory
    results_dir = os.path.join(output_dir, job_id, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Train the model
    train_orientation_model(
        dataset_path=os.path.abspath(dataset_path),
        epochs=epochs,
        batch_size=batch_size,
        output_dir=os.path.abspath(results_dir)
    )
    
    print(f"[ResNet] Training completed. Results saved to {results_dir}")
    sys.stdout.flush()  # Ensure output is flushed


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train AI models from Flask interface.')
    parser.add_argument('--job_id', required=True, help='Unique ID for the training job.')
    parser.add_argument('--model_type', required=True, choices=['yolo', 'resnet'], 
                       help='Type of model to train (yolo or resnet).')
    parser.add_argument('--epochs', required=True, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', required=True, type=int, help='Batch size for training.')
    parser.add_argument('--dataset_path', required=True, help='Path to the extracted dataset.')
    parser.add_argument('--output_dir', required=True, help='Base directory for all uploads.')
    
    args = parser.parse_args()
    
    print(f"Starting training job {args.job_id}")
    print(f"Model type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        # Route to appropriate training function
        if args.model_type == 'yolo':
            train_yolo(args.job_id, args.epochs, args.batch_size, 
                      args.dataset_path, args.output_dir)
        elif args.model_type == 'resnet':
            train_resnet(args.job_id, args.epochs, args.batch_size, 
                        args.dataset_path, args.output_dir)
        
        print(f"Training job {args.job_id} completed successfully.")
        sys.stdout.flush()
        sys.exit(0)  # Explicit exit with success code
        
    except Exception as e:
        print(f"Error during training: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)  # Explicit exit with error code


if __name__ == '__main__':
    main()

