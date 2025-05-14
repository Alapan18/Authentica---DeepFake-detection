import os
from flask import Flask, jsonify, request, send_from_directory, render_template
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Configure Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size

# Configure video directory - use the exact path you specified
VIDEO_FOLDER = 'C:/Users/alapa/Downloads/Authentica---DeepFake-detection/flask_app/video'
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'mov', 'avi', 'mkv'}

# Ensure video folder exists
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve the main application
@app.route('/')
def index():
    return render_template('user_interface5.1.html')

# API endpoint to get list of videos
@app.route('/get_videos')
def get_videos():
    videos = []
    try:
        for file in os.listdir(VIDEO_FOLDER):
            if allowed_file(file):
                videos.append({
                    'name': file,
                    'path': f'/video/{file}'
                })
        return jsonify({'videos': videos})
    except Exception as e:
        logger.error(f"Error fetching videos: {str(e)}")
        return jsonify({'error': 'Failed to fetch videos', 'message': str(e)}), 500

# Serve video files
@app.route('/video/<filename>')
def serve_video(filename):
    return send_from_directory(VIDEO_FOLDER, filename)

# Handle video uploads
@app.route('/upload_videos', methods=['POST'])
def upload_videos():
    if 'videos' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video files found in request'}), 400
    
    uploaded_files = request.files.getlist('videos')
    new_videos = []
    error_files = []
    
    for file in uploaded_files:
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                
                # Add timestamp to filename to avoid duplicates
                name, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                unique_filename = f"{name}_{timestamp}{ext}"
                
                file_path = os.path.join(VIDEO_FOLDER, unique_filename)
                
                # Save the file to the specified directory
                file.save(file_path)
                
                logger.info(f"File uploaded successfully: {unique_filename}")
                
                new_videos.append({
                    'name': unique_filename,
                    'path': f'/video/{unique_filename}'
                })
            except Exception as e:
                logger.error(f"Error saving file {file.filename}: {str(e)}")
                error_files.append(file.filename)
        else:
            error_files.append(file.filename)
            
    if error_files:
        logger.warning(f"Some files couldn't be uploaded: {', '.join(error_files)}")
        
    return jsonify({
        'status': 'success' if new_videos else 'partial' if error_files else 'error',
        'newVideos': new_videos,
        'errors': error_files
    })

if __name__ == '__main__':
    logger.info(f"Starting server with video directory: {VIDEO_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)