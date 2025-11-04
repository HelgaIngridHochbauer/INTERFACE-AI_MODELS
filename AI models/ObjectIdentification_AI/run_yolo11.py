import os
import argparse
from pathlib import Path
from ultralytics import YOLO

try:
    import yaml
except ImportError:
    # Fallback if PyYAML is not installed
    try:
        import ruamel.yaml as yaml
    except ImportError:
        raise ImportError("Please install PyYAML: pip install PyYAML")


def create_yolo_config(dataset_path, output_dir):
    """
    Create a YOLO dataset configuration file.
    Expects dataset structure: dataset_path/train/images, dataset_path/val/images, etc.
    """
    config_path = os.path.join(output_dir, 'dataset_config.yaml')
    
    # Check for common dataset structures
    train_path = None
    val_path = None
    
    # Try to find train and val directories
    # Check top-level first
    try:
        top_level_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    except (OSError, PermissionError):
        top_level_dirs = []
    
    if 'train' in top_level_dirs or 'training' in top_level_dirs:
        train_dir = 'train' if 'train' in top_level_dirs else 'training'
        train_path = train_dir
        # Check if train folder has images subfolder
        train_full_path = os.path.join(dataset_path, train_dir)
        try:
            if os.path.isdir(train_full_path) and 'images' in os.listdir(train_full_path):
                train_path = os.path.join(train_dir, 'images')
        except (OSError, PermissionError):
            pass  # Use train_dir as is
    
    if 'val' in top_level_dirs or 'validation' in top_level_dirs:
        val_dir = 'val' if 'val' in top_level_dirs else 'validation'
        val_path = val_dir
        # Check if val folder has images subfolder
        val_full_path = os.path.join(dataset_path, val_dir)
        try:
            if os.path.isdir(val_full_path) and 'images' in os.listdir(val_full_path):
                val_path = os.path.join(val_dir, 'images')
        except (OSError, PermissionError):
            pass  # Use val_dir as is
    
    # If not found at top level, search recursively (but limit depth)
    if train_path is None or val_path is None:
        for root, dirs, files in os.walk(dataset_path):
            depth = root[len(dataset_path):].count(os.sep)
            if depth > 2:  # Limit search depth
                continue
            if train_path is None and ('train' in dirs or 'training' in dirs):
                train_dir = 'train' if 'train' in dirs else 'training'
                train_path = os.path.relpath(os.path.join(root, train_dir), dataset_path)
            if val_path is None and ('val' in dirs or 'validation' in dirs):
                val_dir = 'val' if 'val' in dirs else 'validation'
                val_path = os.path.relpath(os.path.join(root, val_dir), dataset_path)
            if train_path is not None and val_path is not None:
                break
    
    # If no train/val found, assume the dataset_path itself contains images
    if train_path is None:
        # Check if there are subdirectories that might be classes
        subdirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        if subdirs:
            # Assume first level contains train/val or class folders
            train_path = dataset_path
            val_path = dataset_path
    
    # Create config file
    config = {
        'path': dataset_path,
        'train': train_path if train_path else 'train/images',
        'val': val_path if val_path else 'val/images',
        'nc': 3,  # Default number of classes, can be detected automatically
        'names': ['class1', 'class2', 'class3']  # Default names
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def train_yolo_model(dataset_path, epochs, batch_size, output_dir):
    """
    Train YOLO model with given parameters.
    
    Args:
        dataset_path: Path to the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save results
    """
    # Load a pretrained YOLO11n model (or use 'yolo11n.pt' for base model)
    # Ultralytics will download yolo11n.pt automatically if not found locally
    model_path = "yolo11n.pt"  # Use base model, or specify path to pretrained weights
    
    # Check for existing pretrained weights in the current directory
    pretrained_path = os.path.join(os.getcwd(), "runs", "detect", "train9", "weights", "best.pt")
    if os.path.exists(pretrained_path):
        model_path = pretrained_path
    
    model = YOLO(model_path)
    
    # Create dataset configuration
    config_path = create_yolo_config(dataset_path, output_dir)
    
    # Detect device (GPU if available, otherwise CPU)
    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    # Train the model
    print(f"Starting YOLO training on device: {device}")
    train_results = model.train(
        data=config_path,  # Path to dataset configuration file
        epochs=epochs,
        imgsz=224,  # Image size for training
        batch=batch_size,
        device=device,  # Auto-detect device
        project=output_dir,
        name='yolo_training',
        exist_ok=True,
        verbose=True
    )
    
    print("Training completed, evaluating model...")
    # Evaluate the model's performance on the validation set
    try:
        metrics = model.val()
        print(f"Validation completed. Metrics: {metrics}")
    except Exception as e:
        print(f"Warning: Validation failed: {e}. Continuing...")
    
    # Copy best model to results directory
    best_model_path = os.path.join(output_dir, 'yolo_training', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        final_model_path = os.path.join(output_dir, 'best.pt')
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"Best model saved to {final_model_path}")
    else:
        # Try to find last.pt if best.pt doesn't exist
        last_model_path = os.path.join(output_dir, 'yolo_training', 'weights', 'last.pt')
        if os.path.exists(last_model_path):
            final_model_path = os.path.join(output_dir, 'best.pt')
            import shutil
            shutil.copy2(last_model_path, final_model_path)
            print(f"Last model saved to {final_model_path}")
    
    print("YOLO training function completed successfully.")
    return train_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    train_yolo_model(args.dataset_path, args.epochs, args.batch_size, args.output_dir)

