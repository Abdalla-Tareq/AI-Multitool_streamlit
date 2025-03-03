import streamlit as st
import cv2
import random
from ultralytics import YOLO
import whisper
import os
from pydub import AudioSegment
import torch
import torchvision.transforms as transforms
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import numpy as np

# Configure paths
AudioSegment.ffmpeg = r"C:\PATH\ffmpeg.exe"

upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"

os.makedirs(upload_path, exist_ok=True)
os.makedirs(download_path, exist_ok=True)
os.makedirs(transcript_path, exist_ok=True)

audio_tags = {'comments': 'Converted using pydub!'}

# Object Detection Function
def detect_objects(uploaded_file, model_path):
    if uploaded_file is not None:
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.getvalue())
        input_path = "temp_file"

        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        try:
            if uploaded_file.type.startswith("image"):
                img = cv2.imread(input_path)
                results = model(img)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                        conf = box.conf[0]
                        cls = int(box.cls[0])
                        class_name = result.names[cls]

                        color = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, channels="RGB", caption="Processed Image", use_column_width=True)
            elif uploaded_file.type.startswith("video"):
                st.video(input_path)
            else:
                st.error("Unsupported file type. Please upload an image or video.")
        except Exception as e:
            st.error(f"Error during object detection: {e}")
    else:
        st.warning("Please upload a file.")

# Audio Processing Functions
@st.cache_data(show_spinner=True)
def to_mp3(audio_file, output_audio_file):
    try:
        ext = audio_file.name.split('.')[-1].lower()
        input_path = os.path.join(upload_path, audio_file.name)
        output_path = os.path.join(download_path, output_audio_file)
        audio_data = None

        if ext == "wav":
            audio_data = AudioSegment.from_wav(input_path)
        elif ext == "mp3":
            audio_data = AudioSegment.from_mp3(input_path)
        elif ext == "ogg":
            audio_data = AudioSegment.from_ogg(input_path)
        elif ext == "wma":
            audio_data = AudioSegment.from_file(input_path, "wma")
        elif ext == "aac":
            audio_data = AudioSegment.from_file(input_path, "aac")
        elif ext == "flac":
            audio_data = AudioSegment.from_file(input_path, "flac")
        elif ext == "flv":
            audio_data = AudioSegment.from_flv(input_path)
        elif ext == "mp4":
            audio_data = AudioSegment.from_file(input_path, "mp4")
        else:
            raise ValueError("Unsupported file format!")

        audio_data.export(output_path, format="mp3", tags=audio_tags)
        return output_audio_file
    except Exception as e:
        st.error(f"Error while converting to MP3: {e}")
        return None

@st.cache_resource(show_spinner=True)
def process_audio(filename, model_type):
    try:
        model = whisper.load_model(model_type)
        result = model.transcribe(filename)
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

@st.cache_data(show_spinner=True)
def save_transcript(transcript_data, txt_file):
    try:
        with open(os.path.join(transcript_path, txt_file), "w") as f:
            f.write(transcript_data)
    except Exception as e:
        st.error(f"Error while saving transcript: {e}")

# Image and Video Colorization Functions
def colorize_image(image_path):
    model = torch.load(r"D:\python\colorization_model.pth", map_location=torch.device("cpu"))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        colorized = model(img_tensor)

    colorized_img = colorized.squeeze(0).permute(1, 2, 0).numpy()
    colorized_img = (colorized_img * 255).astype(np.uint8)
    return colorized_img

def colorize_video(video_path):
    clip = VideoFileClip(video_path)

    def process_frame(frame):
        frame_tensor = transforms.ToTensor()(Image.fromarray(frame)).unsqueeze(0)
        with torch.no_grad():
            colorized_frame = colorize_image.model(frame_tensor)
        return np.array(colorized_frame.squeeze(0).permute(1, 2, 0) * 255, dtype=np.uint8)

    colorized_clip = clip.fl_image(process_frame)
    output_path = "colorized_video.mp4"
    colorized_clip.write_videofile(output_path, codec="libx264")
    return output_path

# Main App
st.title("AI Multitool: Object Detection, Speech Recognition & Colorization")

mode = st.sidebar.radio("Choose a Task", ["Object Detection", "Speech Recognition", "Image/Video Colorization"])

if mode == "Object Detection":
    st.header("Object Detection")
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])
    model_path = st.text_input("Enter YOLO model path", "yolov9c.pt")

    if st.button("Detect Objects"):
        detect_objects(uploaded_file, model_path)

elif mode == "Speech Recognition":
    st.header("Speech Recognition")
    st.info('Supports formats - WAV, MP3, MP4, OGG, WMA, AAC, FLAC, FLV')
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "wma", "aac", "flac", "mp4", "flv"])

    if uploaded_file is not None:
        input_path = os.path.join(upload_path, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        output_audio_file = uploaded_file.name.split('.')[0] + '.mp3'
        output_audio_file = to_mp3(uploaded_file, output_audio_file)

        if output_audio_file:
            audio_file_path = os.path.join(download_path, output_audio_file)

            if os.path.exists(audio_file_path):
                st.audio(audio_file_path)

                whisper_model_type = st.radio("Choose Whisper Model", ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

                if st.button("Generate Transcript"):
                    transcript = process_audio(audio_file_path, whisper_model_type.lower())
                    if transcript:
                        output_txt_file = output_audio_file.split('.')[0] + ".txt"
                        save_transcript(transcript, output_txt_file)

                        transcript_path_full = os.path.join(transcript_path, output_txt_file)
                        with open(transcript_path_full, "r") as file:
                            transcript_data = file.read()

                        st.text_area("Transcript", transcript_data, height=300)
                        st.download_button("Download Transcript", transcript_data, file_name=output_txt_file)
                    else:
                        st.error("Failed to generate transcript!")

elif mode == "Image/Video Colorization":
    st.header("Image and Video Colorization")
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        input_path = os.path.join(upload_path, uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.type.startswith("image"):
            st.image(input_path, caption="Uploaded Image", use_column_width=True)
            if st.button("Colorize Image"):
                colorized_img = colorize_image(input_path)
                st.image(colorized_img, caption="Colorized Image", use_column_width=True)

        elif uploaded_file.type.startswith("video"):
            st.video(input_path)
            if st.button("Colorize Video"):
                colorized_video_path = colorize_video(input_path)
                st.video(colorized_video_path, format="video/mp4")
