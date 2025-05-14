#imports
from flask import Flask, request, url_for, redirect, render_template, jsonify, send_from_directory
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
from torch.autograd import Variable
import sys
import random
from torch import nn
from torchvision import models
from torchvision.models import ResNeXt50_32X4D_Weights
import glob
import face_recognition
import shutil
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from moviepy.editor import VideoFileClip
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

# Configure Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size

# Configure video directory
VIDEO_FOLDER = 'C:/Users/alapa/Downloads/Authentica---DeepFake-detection/flask_app/video'
AUDIO_FOLDER = 'C:/Users/alapa/Downloads/Authentica---DeepFake-detection/flask_app/audio'
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'mov', 'avi', 'mkv'}

# Ensure video and audio folders exist
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global lists for predictions and confidences
predictions = []
confidences = []

#Model architecture declaration
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm,_ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Image to tensor configuration
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1, 1, 1], std))

def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png', image*255)
    return image

# Deepfake prediction function
def predict(model, img, path='./'):
    model.eval()
    predictions.clear()
    confidences.clear()
    
    if not isinstance(img, torch.Tensor) or img.shape[0] == 0:
        print("Warning: Empty or invalid input image tensor.")
        return None
    
    # Get dimensions
    batch_size, seq_length, c, h, w = img.shape
    logger.info(f"Processing sequence of {seq_length} frames")
    
    try:
        # Process each frame individually as in original code
        for i in range(seq_length):
            # Process one frame at a time
            frame = img[:, i, :, :, :].unsqueeze(0)
            frame = frame.to('cuda')
            
            fmap, logits = model(frame)
            logits = torch.nn.functional.softmax(logits, dim=1)
            _, prediction = torch.max(logits, 1)
            confidence = logits[:, int(prediction.item())].item() * 100
            
            if int(prediction.item()) == 1:
                predictions.append(1)
            else:
                predictions.append(0)
            confidences.append(confidence)
            
            # Log every single frame instead of every 10th
            logger.info(f'Frame {i+1}/{seq_length}: Prediction = {prediction.item()}, Confidence = {confidence:.2f}%')
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    logger.info(f"Processed total of {len(predictions)} frames")
    return [predictions, confidences]

# Frame extraction for video analysis
class validation_dataset(Dataset):
    def __init__(self, video_names, transform=None):
        self.video_names = video_names
        self.transform = transform
        
    def __len__(self):
        return len(self.video_names)
        
    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        
        # Extract all frames from the video without limits
        for frame in self.frame_extract(video_path):
            if self.transform:
                frames.append(self.transform(frame))
                
        # Ensure we have at least one frame
        if not frames:
            raise ValueError(f"No frames could be extracted from {video_path}")
            
        # Stack all frames into a tensor
        frames = torch.stack(frames)
        return frames.unsqueeze(0)
        
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        # Get total frames for logging
        total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidObj.get(cv2.CAP_PROP_FPS)
        logger.info(f"Processing video with {total_frames} total frames at {fps} FPS")
        
        success = 1
        frame_count = 0
        while success:
            success, image = vidObj.read()
            if success:
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Extracted {frame_count}/{total_frames} frames")
                yield image
        
        logger.info(f"Finished extracting {frame_count} frames from video")


# Image transformation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the deepfake detection model
model = Model(2).cuda()
path_to_model = 'C:/Users/alapa/Downloads/Authentica---DeepFake-detection/models/train_model_99.51_epoch_154.pt'
model.load_state_dict(torch.load(path_to_model,weights_only=False))
model.eval()

# Calculate percentage of real frames
def percent(a):
    frames = len(a)
    if frames == 0:
        return 0
    count = 0
    for k in range(0, frames):
        if a[k] == 1:  # Assuming 1 means 'real'
            count = count + 1
    percentt = (count/frames)*100
    print(frames, " ", count)
    return percentt

# Extract audio from video
def audioslice(a, b):
    c = os.path.join(AUDIO_FOLDER, f"{b}.wav")
    try:
        video = VideoFileClip(a)
        if video.audio is None:
            print(f"Warning: No audio found in video {a}.")
            return None
        video.audio.write_audiofile(c, codec='pcm_s16le')
    except Exception as e:
        print(f"Error extracting audio from video {a}: {e}")
        return None
    return c

# Extract MFCC features from audio
def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

# Analyze audio for deepfake detection
def analyze_audio(input_audio_path):
    model_filename = "C:/Users/alapa/Downloads/Authentica---DeepFake-detection/flask_app/XGBoost_model.pkl"
    scaler_filename = "C:/Users/alapa/Downloads/Authentica---DeepFake-detection/flask_app/CNN_scaler.pkl"

    if input_audio_path is None:
        return "No audio detected in the video."

    if not os.path.exists(input_audio_path):
        return "Error: The specified file does not exist."
    elif not input_audio_path.lower().endswith(".wav"):
        return "Error: The specified file is not a .wav file."

    try:
        mfcc_features = extract_mfcc_features(input_audio_path)
        if mfcc_features is not None:
            scaler = joblib.load(scaler_filename)
            mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

            svm_classifier = joblib.load(model_filename)
            prediction = svm_classifier.predict(mfcc_features_scaled)

            if prediction[0] == 0:
                return "REAL"
            else:
                return "FAKE"
        else:
            return "Unable to process the input audio."
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return "Error occurred during audio analysis."

# Clear global lists
def clear():
    predictions.clear()
    confidences.clear()

'''# Clear folder contents
def clear_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate through the files and subdirectories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file or directory and remove accordingly
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and all its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The folder {folder_path} does not exist.')
'''
# Serve the main application
@app.route('/')
def index():
    return render_template('user_interface5.2.html')

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

# New endpoint to analyze videos for deepfake detection
@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if not request.json or 'videoPath' not in request.json:
        return jsonify({'status': 'error', 'message': 'No video path provided'}), 400
    
    video_path = request.json['videoPath']
    
    # Extract the filename from the path
    filename = os.path.basename(video_path)
    
    # Construct the full path to the video file
    full_path = os.path.join(VIDEO_FOLDER, filename)
    
    if not os.path.exists(full_path):
        return jsonify({
            'status': 'error', 
            'message': f'Video file not found: {filename}'
        }), 404
    
    try:
        # Get video info for the response
        cap = cv2.VideoCapture(full_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        logger.info(f"Starting analysis of video: {filename}, FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Prepare for video analysis - using all frames as in original code
        path_to_videos = [full_path]
        video_dataset = validation_dataset(path_to_videos, transform=train_transforms)
        
        # Process all frames individually
        p = predict(model, video_dataset[0], './')
        
        # Extract audio and analyze
        base_name = os.path.splitext(filename)[0]
        audio_path = audioslice(full_path, base_name)
        audio_result = analyze_audio(audio_path) if audio_path else "No audio detected"
        
        if p is not None:
            preds = p[0]
            confs = p[1]
            
            sum_confidence = 0
            for i in range(len(preds)):
                if preds[i] == 1:
                    sum_confidence += confs[i]
                else:
                    sum_confidence += 100 - confs[i]
            
            percentage = percent(preds)
            avg_confidence = sum_confidence / len(preds) if preds else 0
            
            result = {
                'status': 'success',
                'filename': filename,
                'result': 'fake' if avg_confidence < 50 else 'real',
                'confidence': 100 - avg_confidence if avg_confidence < 50 else avg_confidence,
                'reality_percentage': percentage,
                'audio_result': audio_result,
                'frames_analyzed': len(preds),
                'total_frames': total_frames,
                'fps': fps,
                'duration': f"{int(duration // 60):02d}:{int(duration % 60):02d}",
                'detailed_message': f"{'Fake' if avg_confidence < 50 else 'Real'} | Confidence: {(100 - avg_confidence if avg_confidence < 50 else avg_confidence):.2f}% | Analyzed {len(preds)}/{total_frames} frames | Percentage of reality: {percentage:.2f}% | Audio: {audio_result}"
            }
            
            return jsonify(result)
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to analyze video frames'
            }), 500
            
    except Exception as e:
        logger.error(f"Error analyzing video {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error analyzing video: {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info(f"Starting server with video directory: {VIDEO_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)