import whisper
import torch
from pydub import AudioSegment
import streamlit as st
import spacy
import os

# Cargar modelo de NLP en inglés
nlp = spacy.load("en_core_web_sm")

# Detectar si hay GPU disponible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Diccionario de clasificación en inglés
categories = {
    "disinterest": ["not interested", "don't want", "not for me"],
    "transfer": ["was transferred", "spoke to another agent"],
    "information_request": ["want to know more", "interested", "explain better"]
}

# Convertir MP3 a WAV con mejora en calidad
def convert_to_wav(mp3_path):
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"File not found: {mp3_path}")

    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Improve clarity
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    
    return wav_path

# Cargar Whisper en GPU si está disponible
model = whisper.load_model("medium").to(device)


# Transcribir con Whisper en inglés
def transcribe_audio(mp3_path):
    wav_path = convert_to_wav(mp3_path)  # Convert to WAV
    result = model.transcribe(wav_path)
    return result["text"]

# Clasificación con NLP
def classify_call(text):
    text = text.lower()
    doc = nlp(text)
    
    for category, keywords in categories.items():
        if any(word in text for word in keywords):
            return category
    return "undefined"

# Interfaz con Streamlit
st.title("Call Quality Analysis with Whisper (Optimized with GPU)")

audio_file = st.file_uploader("Upload an MP3 audio file", type=["mp3"])
if audio_file:
    with open(audio_file.name, "wb") as f:
        f.write(audio_file.getbuffer())  # Save temporary file
    
    transcribed_text = transcribe_audio(audio_file.name)
    classification = classify_call(transcribed_text)

    st.write("**Transcription:**", transcribed_text)
    st.write("**Classification:**", classification)
    st.write("Using device:", device)  # Verificar si usa la GPU
