# 🛡️ Authentica Deepfake Detection System  

A **multimodal deepfake detection system** leveraging **CNN-LSTM for video analysis** and **SVM-based classification for audio detection**. The project includes training, prediction, and a web-based interface for  deepfake detection.  

---

###  **Prototype**

[Watch the video on YouTube](https://youtu.be/hvwSe1kOR3c)


## 📂 Project Structure
```bash
│
├───audio final
│   └───audio_train_test_code.ipynb
├───video final
│   ├───predict.ipynb
│   ├───helper.ipynb
│   └───test.ipynb
├───graph
├───flask_app
│   ├───app.py
│   ├───scaler.pkl
│   ├───svm_model.pkl
│   ├───test.ipynb
│   ├───static
│   │   ├───deepfake.png
│   │   ├───loader.gif
│   │   └───styles
│   │           stylesdeepfake.css
│   ├───templates
│   │   ├───index.html
│   │   └───web.html
│   └───video
├───testcases
├───README.md
└───requirements.txt
```

### 🔊 **`audio_final/`**  
📌 Contains the **training and prediction code** for **audio deepfake detection** using **MFCC feature extraction and scaling with XGBoost classifier ** and **Mel-Spectogram genaration and normalization with CNN classifier**.  

### 🎥 **`video_final/`**  
📌 Contains the **training and prediction code** for **video-based deepfake detection** using a **CNN-LSTM architecture**.  
📌 Leverages **ResNeXt50 for feature extraction** and an **LSTM model** to analyze temporal inconsistencies in videos.  

### 🧪 **`testcases/`**  
📌 Includes **sample testing videos and audio files** for evaluating the deepfake detection models.  
📌 Helps verify system performance before deployment.  

### 🌐 **`flask_app/`**  
📌 Contains the **Flask web application** for hosting the deepfake detection system.  
📌 Includes:  
- **HTML files** for the frontend UI.  
- **CSS stylesheets** for styling the web pages.  
- **JavaScript & Flask API** for deepfake detection.  
- **Images and assets** for UI design.  

---

## 🚀 How to Use  

## 📂 **Dataset**

 - CELEB DF V1- https://drive.google.com/open?id=10NGF38RgF8FZneKOuCOdRIsPzpC7_WDd
 - CELEB DF V2 - https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj
 - DFDC dataset: https://www.kaggle.com/c/deepfake-detection-challenge/data
 - Face Forensics++: https://www.kaggle.com/datasets/hungle3401/faceforensics
 - Face Forensics: https://github.com/ondyari/FaceForensics

## ⚙️ Pretrained video model

- Drive Link: https://drive.google.com/file/d/1B57pBBVApRiiKSjl-2PFHgd5b4Sj3RNL/view?usp=sharing



### 🛠️ **Installation**  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/Alapan18/Authentica---DeepFake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

This project includes multiple components for **deepfake detection**. Below is a breakdown of key files and their execution commands.  

### 🔊 **Audio Deepfake Detection**  
📌 **Files in `audio_final/`**  
- `train_test_audio.py` → Trains and tests the **audio deepfake detection model** using MFCC features.  
- `scaler.pkl` → Stores the **standard scaler** used to normalize audio data.  
- `svm_model.pkl` → Pretrained **SVM model** for deepfake audio detection.  

**Run Training & Prediction:**  
```bash
cd audio_final
python train_test_audio.py --train  # Train and test the audio model in the bash shell only.
```
### 🎥 **Video Deepfake Detection**  

📌 **Files in `video_final/`**  
- `predict.ipynb` → Jupyter Notebook for **video deepfake detection** using a pretrained CNN-LSTM model.  
- `test.ipynb` → Notebook for **testing the trained model** on video samples.  

**Run Video Prediction (Jupyter Notebook):**  
```bash
cd video_final
jupyter notebook predict.ipynb
```
### 🌐 **Flask Web Application**  

📌 **Files in `flask_app/`**  
- `app.py` → **Flask backend** to host the web-based deepfake detection system.  
- `scaler.pkl` & `svm_model.pkl` → The **audio deepfake detection models** used in the backend.  
- `test.ipynb` → Jupyter Notebook for **debugging Flask API requests**.  
- `static/` → Contains UI assets like **CSS (`stylesdeepfake.css`)**, images (`deepfake.png`), and loader animations (`loader.gif`).  
- `templates/` → HTML pages for the **web interface** (`index.html`, `web.html`).  
- `video/` → Stores processed video data for Flask API.  

**Run Flask Web App:**  
```bash
cd flask_app
python app.py
```

## Results for Different Dataset Sizes and Epochs

| Dataset Size | Real and Fake Count | Epochs | Training Acc | Testing Acc |
|--------------|---------------------|--------|--------------|-------------|
| 2 GB         | R: 408  F: 795       | 50     | 94.39%       | 81.06%      |
| 10 GB        | R: 890  F: 5639      | 20     | 94.66%       | 86.76%      |
| 10 GB        | R: 890  F: 5639      | 40     | 97.49%       | 90.65%      |

### 🙋‍♂️ **Authors**

-  Alapan Pradhan | alapan.pradhan.1@gmail.com
-  Samir Roy | roysamir929@gmail.com
-  Adrith Ray | ayushray268@gmail.com
-  Dhrubojyoti Saha | dhrubojyoti2003saha@gmail.com
-  Sagarika Deb | sagarikadeb498@gmail.com
<br>
<br>
<br>


<p align= "center"><font size="40">THANK YOU</font></p>
