# **AI Multitool: Object Detection, Speech Recognition & Image/Video Colorization** 🎯  

This project is a **web-based AI tool** that provides multiple functionalities for processing images, videos, and audio files. It includes:  

✅ **Object Detection** using YOLO  
✅ **Speech Recognition** using Whisper  
✅ **Image & Video Colorization** using a deep learning model  

The application is built with **Streamlit** as the user interface and utilizes AI libraries such as **PyTorch, OpenCV, MoviePy, Whisper, YOLO, and pydub**.

---

## **🚀 Features**
- **Object Detection**:
  - Detects objects in images and videos using **YOLO**.
  - Draws bounding boxes with labels and confidence scores.
  - Supports **JPG, PNG, MP4, AVI, MOV** formats.

- **Speech Recognition**:
  - Converts **audio files to text** using **Whisper AI**.
  - Supports formats like **MP3, WAV, MP4, OGG, AAC, FLAC, FLV**.
  - Allows choosing the **Whisper model size** for better performance.

- **Image & Video Colorization**:
  - Colorizes **grayscale images and videos** using a deep learning model.
  - Processes videos **frame-by-frame** using `MoviePy`.
  - Supports **JPG, PNG, MP4** formats.

---

## **🛠️ Technologies Used**
- **Python 3.12**
- **Streamlit** - Web framework
- **PyTorch** - Deep learning
- **OpenCV, Pillow** - Image processing
- **MoviePy** - Video processing
- **Whisper, pydub** - Speech recognition
- **YOLO** - Object detection  

---

## **📂 Project Structure**
- **├── app.py                    # Main Streamlit application**
- **├── uploads/                  # Stores user-uploaded files**
- **├── downloads/                # Stores processed files**
- **├── transcripts/              # Stores transcribed text files**
- **├── models/                   # Stores AI models (YOLO & Colorization)**
- **└── README.md                 # Project documentation**


---

## **📊 How to Run**
1. **Install dependencies:**  
   ```bash
   pip install streamlit opencv-python ultralytics whisper pydub torch torchvision numpy moviepy pillow

2. **Run the application:**
   - streamlit run app.py

## 📊 Results & Performance
**Object Detection: Displays detected objects with bounding boxes and confidence scores.**
**Speech Recognition: Provides accurate text transcriptions of uploaded audio files.**
**Colorization: Generates high-quality colorized images and videos from grayscale inputs.**

##🤝 Contributing
**🚀 Contributions are welcome! You can:**
- ✅ Improve AI models for better performance.
- ✅ Optimize file handling and processing.
- ✅ Add new features such as face detection or voice-to-text translation improvements.

##👤 Developer
##👨‍💻 Abdallah Tarek
- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/abdalla-tarek-21a025263/)


🚀 A powerful AI toolkit for processing images, videos, and audio files!
