import os
import uuid
import zipfile
import threading
import subprocess
import json
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for, flash, redirect

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


# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def update_status(job_id, status, message=""):
    """Updates the status JSON file for a given job."""
    status_file = os.path.join(app.config['UPLOAD_FOLDER'], job_id, 'status.json')
    with open(status_file, 'w') as f:
        json.dump({'status': status, 'message': message}, f)

def start_training_thread(job_id, model_type, epochs, batch_size, dataset_path):
    """
    This function is executed in a separate thread to run the training script.
    It prevents the web request from timing out during long training sessions.
    """
    try:
        # Update status to 'training'
        update_status(job_id, 'training', 'The model training process has started.')

        # Call the actual training script using subprocess.
        # This runs the training in a separate process.
        # We pass all necessary parameters as command-line arguments.
        command = [
            'python',
            'train.py',
            '--job_id', job_id,
            '--model_type', model_type,
            '--epochs', str(epochs),
            '--batch_size', str(batch_size),
            '--dataset_path', dataset_path,
            '--output_dir', app.config['UPLOAD_FOLDER']
        ]
        
        # Execute the command. The `check=True` will raise an exception if the script fails.
        subprocess.run(command, check=True, capture_output=True, text=True)

        # If the script completes successfully, update status to 'completed'.
        update_status(job_id, 'completed', 'Training finished successfully. Your model is ready for download.')

    except subprocess.CalledProcessError as e:
        # If the training script fails, capture the error and update the status.
        error_message = f"Training failed. Error: {e.stderr}"
        update_status(job_id, 'failed', error_message)
    except Exception as e:
        # Catch any other unexpected errors.
        update_status(job_id, 'failed', f"An unexpected error occurred: {str(e)}")


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
        return redirect(request.url)

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

    # Create the initial status file.
    update_status(job_id, 'pending', 'Your job has been queued.')

    # Unzip the dataset.
    dataset_path = os.path.join(job_folder, 'dataset')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)

    # Get training parameters from the form.
    model_type = request.form['model']
    epochs = int(request.form['epochs'])
    batch_size = int(request.form['batch'])

    # Start the training process in a background thread.
    thread = threading.Thread(
        target=start_training_thread,
        args=(job_id, model_type, epochs, batch_size, dataset_path)
    )
    thread.start()

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

    # Create a zip file of the results folder.
    output_zip_path = os.path.join(job_folder, f'model_{job_id}.zip')
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

