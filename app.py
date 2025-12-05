import os
import uuid
import zipfile
import threading
import subprocess
import json
import smtplib
import shutil
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for, flash, redirect
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Create an 'uploads' directory if it doesn't exist. This will store datasets and trained models.
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define allowed file extensions for security.
ALLOWED_EXTENSIONS = {'zip'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# A secret key is needed for flashing messages.
app.config['SECRET_KEY'] = os.urandom(24)

# Email configuration - these can be set via environment variables
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
app.config['MAIL_FROM'] = os.environ.get('MAIL_FROM', app.config['MAIL_USERNAME'])


# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_dataset_structure(dataset_path, model_type):
    """
    Validates that the dataset has the correct structure for the specified model type.
    Returns (is_valid, error_message)
    """
    if not os.path.exists(dataset_path):
        return False, "Dataset path does not exist."
    
    try:
        items = os.listdir(dataset_path)
        if not items:
            return False, "Dataset folder is empty."
    except (OSError, PermissionError):
        return False, "Cannot access dataset folder."
    
    if model_type == 'yolo':
        # YOLO expects train/val structure with images and labels
        has_train = False
        has_val = False
        
        for item in items:
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                if item.lower() in ['train', 'training']:
                    has_train = True
                elif item.lower() in ['val', 'validation', 'test']:
                    has_val = True
        
        if not has_train:
            return False, "YOLO dataset must contain a 'train' or 'training' folder."
        if not has_val:
            return False, "YOLO dataset must contain a 'val', 'validation', or 'test' folder."
        
        return True, ""
    
    elif model_type == 'resnet':
        # ResNet expects images with corresponding angle/label files
        # Check if there are any image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        has_images = False
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    has_images = True
                    break
            if has_images:
                break
        
        if not has_images:
            return False, "ResNet dataset must contain image files (.jpg, .png, etc.)."
        
        return True, ""
    
    return False, f"Unknown model type: {model_type}"

def update_status(job_id, status, message=""):
    """Updates the status JSON file for a given job."""
    try:
        status_file = os.path.join(app.config['UPLOAD_FOLDER'], job_id, 'status.json')
        job_folder = os.path.dirname(status_file)
        
        # Ensure job folder exists
        if not os.path.exists(job_folder):
            print(f"[ERROR] Job folder does not exist: {job_folder}")
            os.makedirs(job_folder, exist_ok=True)
        
        # Read existing status to preserve email if it exists
        existing_data = {}
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[WARNING] Could not read existing status file: {e}")
                existing_data = {}
        
        existing_data['status'] = status
        existing_data['message'] = message
        
        with open(status_file, 'w') as f:
            json.dump(existing_data, f)
        
        print(f"[Status] Updated job {job_id} to '{status}': {message}")
    except Exception as e:
        print(f"[ERROR] Failed to update status for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise so caller knows it failed

def send_email_notification(to_email, job_id, model_type):
    """
    Sends an email notification with job ID and instructions on how to download the model.
    This is used instead of sending the zip file as attachment (which might be too large).
    """
    try:
        # Get the absolute path to the uploads folder
        uploads_path = os.path.abspath(app.config['UPLOAD_FOLDER'])
        job_folder = os.path.join(uploads_path, job_id)
        zip_filename = f'model_{job_id}.zip'
        
        # Get base URL from environment or use default
        base_url = os.environ.get('BASE_URL', 'http://localhost:5000')
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = app.config['MAIL_FROM']
        msg['To'] = to_email
        msg['Subject'] = f'Model Training Complete - Job ID: {job_id}'
        
        # Create email body with instructions
        body = f"""Your model training has completed successfully!

Job ID: {job_id}
Model Type: {model_type.upper()}

To download your trained model:

1. You can download it directly from the web interface by visiting:
   {base_url}/download/{job_id}

2. Or navigate to the following folder on the server:
   {job_folder}
   
   Look for the file named: {zip_filename}

The zip file contains all the trained model files and results.

Thank you for using our model training service!
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to server and send email
        server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
        if app.config['MAIL_USE_TLS']:
            server.starttls()
        
        if app.config['MAIL_USERNAME'] and app.config['MAIL_PASSWORD']:
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        
        text = msg.as_string()
        server.sendmail(app.config['MAIL_FROM'], to_email, text)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def start_training_thread(job_id, model_type, epochs, batch_size, dataset_path, email=None):
    """
    This function is executed in a separate thread to run the training script.
    It prevents the web request from timing out during long training sessions.
    """
    try:
        # Immediately update status to 'training' to show progress
        print(f"[Thread] Starting training for job {job_id}")
        try:
            update_status(job_id, 'training', 'The model training process has started.')
            print(f"[Thread] Status updated to 'training' for job {job_id}")
        except Exception as status_error:
            print(f"[Thread] ERROR: Failed to update status to 'training': {status_error}")
            import traceback
            traceback.print_exc()
            # Try to update status to failed
            try:
                update_status(job_id, 'failed', f'Failed to start training: {str(status_error)}')
            except:
                pass
            return  # Exit thread if we can't update status

        # Call the actual training script using subprocess.
        # This runs the training in a separate process.
        # We pass all necessary parameters as command-line arguments.
        import sys
        
        # Get the absolute path to train.py
        train_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')
        
        # Use sys.executable to ensure we use the same Python interpreter
        command = [
            sys.executable,
            train_script_path,
            '--job_id', job_id,
            '--model_type', model_type,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--dataset_path', os.path.abspath(dataset_path),
            '--output_dir', os.path.abspath(app.config['UPLOAD_FOLDER'])
        ]
        
        # Execute the command. Capture output for error parsing
        # We'll check the return code manually to handle errors better
        result = subprocess.run(
            command, 
            check=False,  # Don't raise exception, check returncode instead
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True, 
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Print output to terminal for real-time monitoring
        if result.stdout:
            print(result.stdout)
            sys.stdout.flush()
        
        # Check if training failed
        if result.returncode != 0:
            # Training failed - raise exception with the output
            error_output = result.stdout if result.stdout else f"Process exited with code {result.returncode}"
            raise subprocess.CalledProcessError(result.returncode, command, output=error_output)

        # If the script completes successfully, create the zip file and handle email/download
        job_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        results_folder = os.path.join(job_folder, 'results')
        output_zip_path = os.path.join(job_folder, f'model_{job_id}.zip')
        
        # Create zip file
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            for root, _, files in os.walk(results_folder):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), results_folder)
                    zipf.write(os.path.join(root, file), arcname=relative_path)
        
        # Update status to completed first
        update_status(job_id, 'completed', 'Training finished successfully. Your model is ready for download.')
        
        # If email is provided, send notification email with instructions (not the zip file)
        if email and email.strip():
            email_sent = send_email_notification(
                to_email=email.strip(),
                job_id=job_id,
                model_type=model_type
            )
            if email_sent:
                # Update status to reflect email was sent
                status_file = os.path.join(job_folder, 'status.json')
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                status_data['message'] = 'Training finished successfully. Check your email for download instructions (Job ID: ' + job_id + ').'
                # Keep email in status for reference
                with open(status_file, 'w') as f:
                    json.dump(status_data, f)
            else:
                # Email failed - update message but keep download button available
                status_file = os.path.join(job_folder, 'status.json')
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                status_data['message'] = 'Training finished successfully. Your model is ready for download. (Note: Email notification failed, but you can still download using the button below.)'
                with open(status_file, 'w') as f:
                    json.dump(status_data, f)

    except subprocess.CalledProcessError as e:
        # If the training script fails, capture the error and update the status.
        # Get error output from the exception
        error_output = ""
        if hasattr(e, 'output') and e.output:
            error_output = e.output
        elif hasattr(e, 'stderr') and e.stderr:
            error_output = e.stderr
        elif hasattr(e, 'stdout') and e.stdout:
            error_output = e.stdout
        else:
            error_output = str(e)
        
        # Check if error is related to dataset structure
        error_lower = error_output.lower()
        structure_keywords = ['train', 'validation', 'val', 'folder', 'directory', 'not found', 
                             'missing', 'no such file', 'file not found', 'dataset', 'structure',
                             'images', 'labels', 'yaml', 'config', 'path']
        
        if any(keyword in error_lower for keyword in structure_keywords):
            user_message = "Failed: Input files are incorrect. Please check your dataset structure and try again."
        else:
            user_message = f"Training failed: {error_output[:200]}"  # Limit message length
        
        # Print full error to terminal for debugging
        print(f"\n{'='*60}")
        print(f"Training Error for Job {job_id}:")
        print(f"{'='*60}")
        print(error_output)
        print(f"{'='*60}\n")
        
        update_status(job_id, 'failed', user_message)
        
    except Exception as e:
        # Catch any other unexpected errors.
        error_str = str(e)
        error_lower = error_str.lower()
        
        # Check if error is related to dataset structure
        structure_keywords = ['train', 'validation', 'val', 'folder', 'directory', 'not found',
                             'missing', 'no such file', 'file not found', 'dataset', 'structure']
        
        if any(keyword in error_lower for keyword in structure_keywords):
            user_message = "Failed: Input files are incorrect. Please check your dataset structure and try again."
        else:
            user_message = f"An unexpected error occurred: {error_str[:200]}"
        
        # Print full error to terminal for debugging
        print(f"\n{'='*60}")
        print(f"Unexpected Error for Job {job_id}:")
        print(f"{'='*60}")
        print(f"Error: {error_str}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        
        update_status(job_id, 'failed', user_message)


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles the file upload and initiates the training process."""
    # --- Validation Checks ---
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading.')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a .zip file.')
        return redirect(url_for('index'))

    # --- If validation passes, proceed with the job ---
    # Generate a unique ID for this training job.
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_folder)

    # Save the uploaded zip file.
    zip_path = os.path.join(job_folder, 'dataset.zip')
    file.save(zip_path)

    # Get training parameters from the form.
    model_type = request.form['model']
    epochs = int(request.form['epochs'])
    batch_size = int(request.form['batch'])
    email = request.form.get('email', '').strip()  # Optional email field

    # Create the initial status file (with email if provided)
    if email:
        status_file = os.path.join(job_folder, 'status.json')
        with open(status_file, 'w') as f:
            json.dump({'status': 'pending', 'message': 'Your job has been queued.', 'email': email}, f)
    else:
        update_status(job_id, 'pending', 'Your job has been queued.')

    # Unzip the dataset.
    dataset_path = os.path.join(job_folder, 'dataset')
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
    except zipfile.BadZipFile:
        # Clean up the job folder
        shutil.rmtree(job_folder, ignore_errors=True)
        flash('Invalid zip file. Please upload a valid .zip archive.')
        return redirect(url_for('index'))
    except Exception as e:
        # Clean up the job folder
        shutil.rmtree(job_folder, ignore_errors=True)
        flash(f'Error extracting zip file: {str(e)}')
        return redirect(url_for('index'))

    # Start the training process in a background thread.
    print(f"[Upload] Starting training thread for job {job_id}")
    thread = threading.Thread(
        target=start_training_thread,
        args=(job_id, model_type, epochs, batch_size, dataset_path, email if email else None),
        daemon=True  # Thread will terminate when main program exits
    )
    thread.start()
    print(f"[Upload] Training thread started for job {job_id}")

    # Redirect user to a page where they can monitor the status.
    return redirect(url_for('job_status_page', job_id=job_id))


@app.route('/status/<job_id>')
def job_status_page(job_id):
    """Renders a page that shows the current status of the job."""
    return render_template('status.html', job_id=job_id)


@app.route('/api/status/<job_id>')
def get_job_status(job_id):
    """API endpoint for the frontend to fetch the latest job status."""
    status_file = os.path.join(app.config['UPLOAD_FOLDER'], job_id, 'status.json')
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        # Don't expose email in API response for security
        if 'email' in status_data:
            status_data['has_email'] = bool(status_data.get('email'))
            del status_data['email']
        return jsonify(status_data)
    else:
        return jsonify({'status': 'error', 'message': 'Job ID not found.'}), 404


@app.route('/download/<job_id>')
def download_model(job_id):
    """Allows the user to download the final trained model as a zip archive."""
    job_folder = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    results_folder = os.path.join(job_folder, 'results')
    
    # Check if training is complete and results exist.
    status_file = os.path.join(job_folder, 'status.json')
    if not os.path.exists(status_file):
         flash('Job ID not found.')
         return redirect(url_for('index'))

    with open(status_file, 'r') as f:
        status = json.load(f).get('status')
    
    if status != 'completed':
        flash('Model is not ready for download. Current status: ' + status)
        return redirect(url_for('job_status_page', job_id=job_id))

    # Check if results folder exists
    if not os.path.exists(results_folder):
        flash('Results folder not found. Training may not have completed successfully.')
        return redirect(url_for('job_status_page', job_id=job_id))

    # Create a zip file of the results folder if it doesn't already exist.
    output_zip_path = os.path.join(job_folder, f'model_{job_id}.zip')
    if not os.path.exists(output_zip_path):
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            for root, _, files in os.walk(results_folder):
                for file in files:
                    # Create a relative path for files inside the zip.
                    relative_path = os.path.relpath(os.path.join(root, file), results_folder)
                    zipf.write(os.path.join(root, file), arcname=relative_path)

    # Send the zip file to the user.
    return send_from_directory(directory=job_folder, path=f'model_{job_id}.zip', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

