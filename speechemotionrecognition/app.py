import streamlit as st
import librosa
import numpy as np
from keras.models import load_model

# Dictionary mapping emotions to emojis
emojis = {
    'angry': '😡',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😄',
    'neutral': '😐',
    'sad': '😢',
    'pleasant surprise': '😮'
}
def extract_mfcc(audio_file_path):
    audio, sr = librosa.load(audio_file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_reshaped = np.expand_dims(mfcc_mean.T, axis=0)
    mfcc_reshaped = np.expand_dims(mfcc_reshaped, axis=-1)
    mfcc_reshaped = np.expand_dims(mfcc_reshaped, axis=-1)
    return mfcc_reshaped
st.title("Emotion Recognition from Audio")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
if uploaded_file:
    mfcc_features = extract_mfcc(uploaded_file)
    model = load_model('emotion_detection_model.h5')
    predicted_emotion = model.predict(mfcc_features)
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    predicted_index = np.argmax(predicted_emotion, axis=1)
    predicted_label = class_names[predicted_index[0]]
    emoji = emojis.get(predicted_label, '❓')
    st.write(f"Predicted Emotion: {predicted_label} {emoji}")
else:
    st.write("Please upload an audio file to see your emotion!")
