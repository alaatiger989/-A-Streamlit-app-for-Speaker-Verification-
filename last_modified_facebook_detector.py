import streamlit as st
import sounddevice as sd
import numpy as np
import pandas as pd
from scipy.io.wavfile import write
import scipy.io.wavfile as wavfile
import nemo.collections.asr as nemo_asr
import threading
import time
import os
import io
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import librosa
import torchaudio
# Load the NeMo speaker model
@st.cache_resource
def load_model():
    return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
@st.cache_resource
def audio_deep_fake_detector_model():
    model_name = "facebook/wav2vec2-large-robust-ft-swbd-300h"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    return model , processor
speaker_model = load_model()
deep_fake_detector_model , processor = audio_deep_fake_detector_model()

def detect_deep_fake_audio(audio_path):
    # Load audio file
    audio, sample_rate = librosa.load(audio_path, sr=16000)

    # Process the audio file
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)

    # Get the model's output
    with torch.no_grad():
        logits = deep_fake_detector_model(**inputs).logits

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the label with the highest probability
    predicted_label = torch.argmax(probabilities, dim=-1).item()

    # For this example, let's assume:
    # 0: Real, 1: Fake
    labels = ["Fake" , "Real"]
    st.write(f"The audio is predicted to be: {labels[predicted_label]}")
    return predicted_label


# Function to save audio recording
def save_wav(filename, data, samplerate):
    write(filename, samplerate, data)
# Record audio with a maximum duration of 3 minutes
def record_audio(max_duration=5):
    fs = 16000  # Sample rate
    st.write("Recording... (up to 5 seconds)")
    recording = []

    def callback(indata, frames, time, status):
        recording.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback, dtype=np.int16):
        progress_bar = st.progress(0)
        for i in range(max_duration):
            time.sleep(1)
            progress_bar.progress((i + 1) / max_duration)
        st.write("Recording finished")

    recording = np.concatenate(recording, axis=0)
    return recording[:fs * max_duration], fs

def record_audio_with_duration(duration):
    fs = 16000  # Sample rate
    st.write("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished")
    return myrecording, fs

# Process audio to extract embeddings
def extract_embeddings(file_path):
    embs = speaker_model.get_embedding(file_path)
    return embs.cpu().numpy()

# Function to check if a user ID already exists
def user_id_exists(user_id, filename='embeddings.csv'):
    if os.path.exists(filename):
        saved_embeddings_df = pd.read_csv(filename)
        return user_id in saved_embeddings_df['user_id'].values
    return False

# Save embeddings to a CSV file
def save_embeddings(user_id, embeddings, filename='embeddings.csv'):
    if user_id_exists(user_id, filename):
        st.warning("User ID already exists. Please enter a different user ID.")
        return False
    embeddings_list = embeddings.tolist()
    df = pd.DataFrame({'user_id': [user_id], 'embeddings': [embeddings_list]})
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
    return True

# Verify user based on a new audio recording
def verify_user(user_id, new_embeddings, saved_embeddings_file='embeddings.csv', threshold=0.05):
    saved_embeddings_df = pd.read_csv(saved_embeddings_file)
    saved_embeddings_df['embeddings'] = saved_embeddings_df['embeddings'].apply(eval)
    user_embeddings = saved_embeddings_df[saved_embeddings_df['user_id'] == user_id]['embeddings'].values
    if len(user_embeddings) == 0:
        return None, False
    saved_embeddings = np.stack(user_embeddings)
    similarities = np.dot(saved_embeddings, new_embeddings.T)
    max_similarity = similarities.max()
    return max_similarity, max_similarity > threshold
def handle_recording(duration, filename, user_id):
    recording, samplerate = record_audio_with_duration(duration)
    save_wav(filename, recording, samplerate)
    st.success(f"Audio saved as {filename}")
    embeddings = extract_embeddings(filename)
    if user_id:
        success = save_embeddings(user_id, embeddings)
        predicted_label = detect_deep_fake_audio(filename)
        if predicted_label == 1:
            if success:
                st.success("User enrolled successfully.")
        else:
            st.error("User Cannot enroll with Fake or Ai Generated Audio")
    else:
        st.error("User ID not Entered.")
def start_recording():
    st.session_state.recording = True
    st.session_state.record_button_key = st.session_state.get('record_button_key', 0) + 1
    st.session_state.start_time = sd.query_devices(sd.default.device, 'input')['default_samplerate']
    st.session_state.audio_data = sd.rec(int(60 * st.session_state.start_time), samplerate=st.session_state.start_time,
                                         channels=1, dtype='int16')
    sd.wait()

def stop_recording():
    st.session_state.recording = False
    st.session_state.stop_button_key = st.session_state.get('stop_button_key', 0) + 1


# Function to save audio to a .wav file
def save_audio(recording, fs, filename):
    wavfile.write(filename, fs, recording)
    st.success(f"Audio saved as {filename}")
    return filename

# Function to play audio
def play_audio(audio_data, fs):
    audio_bytes = io.BytesIO()
    wavfile.write(audio_bytes, fs, audio_data)
    st.audio(audio_bytes, format='audio/wav')

# Streamlit UI
st.title("Speaker Verification App")

# Initialize state
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'record_button_key' not in st.session_state:
    st.session_state.record_button_key = 0
if 'stop_button_key' not in st.session_state:
    st.session_state.stop_button_key = 0

# Create tabs
tab3, tab4 = st.tabs(["Enroll", "Verify After Enrollment"])

with tab3:
    st.header("Enroll")
    user_id = st.text_input("Enter your user ID for enrollment")
    enroll_option = st.selectbox("Choose an option for enrollment", ("Upload Audio", "Record Unlimited Audio", "Record Using Specific Duration" , "Record For 5 Seconds"))
    if user_id_exists(user_id):
        st.error("User ID already exists. Please enter a different user ID.")
    else:
        if enroll_option == "Record Unlimited Audio":
            # Record button
            if st.session_state.recording:
                if st.button("Stop" , key = f"enroll_stop_button_{st.session_state.stop_button_key}"):
                    stop_recording()
            else:
                if st.button("Record", key = f"enroll_record_button_{st.session_state.record_button_key}"):
                    start_recording()

            # Play button
            if st.session_state.audio_data is not None:
                if st.button("Play", key = "enroll_play_button"):
                    play_audio(st.session_state.audio_data, int(st.session_state.start_time))

                # Save button
                if st.button("Save", key = "enroll_save_button"):
                    filename = save_audio(st.session_state.audio_data, int(st.session_state.start_time) , "output.wav")
                    embeddings = extract_embeddings('output.wav')
                    predicted_label = detect_deep_fake_audio('output.wav')
                    if predicted_label == 1:
                        if save_embeddings(user_id, embeddings):
                            st.success("User enrolled successfully.")
                    else:
                        st.error("User Cannot enroll with Fake or Ai Generated Audio")
        if enroll_option == "Record For 5 Seconds":
            if st.button("Start Recording for Enrollment"):
                recording, samplerate = record_audio()
                save_wav('enroll5sec.wav', recording, samplerate)
                embeddings = extract_embeddings('enroll5sec.wav')
                predicted_label = detect_deep_fake_audio('enroll5sec.wav')
                if predicted_label == 1:
                    if save_embeddings(user_id, embeddings) :
                        st.success("User enrolled successfully.")
                else:
                    st.error("User Cannot enroll with Fake or Ai Generated Audio")
        if enroll_option == "Record Using Specific Duration":
            duration = st.slider("Select recording duration (seconds)", 1, 10, 3, key="enroll_duration")
            if st.button("Start Recording for Enrollment"):
                handle_recording(duration , 'enroll_specific_duration.wav' , user_id)
                #threading.Thread(target=handle_recording, args=(duration, 'enroll_specific_duration.wav', user_id)).start()
        if enroll_option == "Upload Audio":
            enroll_file = st.file_uploader("Choose a .wav file for enrollment", type="wav", key="enroll_upload")
            if enroll_file is not None:
                with open("enroll.wav", "wb") as f:
                    f.write(enroll_file.getbuffer())
                st.success("File uploaded successfully")

                # Extract and save embeddings for enrollment
                embeddings = extract_embeddings("enroll.wav")
                predicted_label = detect_deep_fake_audio('enroll.wav')
                if predicted_label == 1:
                    if save_embeddings(user_id, embeddings):
                        st.success("User enrolled successfully.")
                else:
                    st.error("User Cannot enroll with Fake or Ai Generated Audio")

with tab4:
    st.header("Verify After Enrollment")
    user_id = st.text_input("Enter your user ID for verification after enrollment")
    verify_option = st.selectbox("Choose an option for verification after enrollment", ("Upload Audio", "Record Unlimited Audio", "Record Using Specific Duration" , "Record For 5 Seconds"))
    if verify_option == "Record Unlimited Audio":
        if user_id_exists(user_id):
            if st.session_state.recording:
                if st.button("Stop", key = f"verify_stop_button_{st.session_state.stop_button_key}"):
                    stop_recording()
            else:
                if st.button("Record", key = f"verify_record_button_{st.session_state.record_button_key}"):
                    start_recording()

            # Play button
            if st.session_state.audio_data is not None:
                if st.button("Play", key = "verify_play_button"):
                    play_audio(st.session_state.audio_data, int(st.session_state.start_time))

                # Save button
                if st.button("Save", key = "verify_stop_button"):
                    filename = save_audio(st.session_state.audio_data, int(st.session_state.start_time) , filename = "verify_after_enrollment.wav")
                    new_embeddings = extract_embeddings('verify_after_enrollment.wav')
                    similarity, is_same_speaker = verify_user(user_id, new_embeddings)
                    number = similarity
                    percentage = number * 1000
                    formatted_percentage = f"{percentage:.2f}%"
                    if percentage < 0:
                        st.write("Similarity score: 0%")
                    elif percentage > 100.00:
                        st.write("Similarity score: 100%")
                    else:
                        st.write(f"Similarity score: {formatted_percentage}")
                    if similarity is not None:

                        if is_same_speaker:
                            st.write("Decision: Verified.")
                        else:
                            st.write("Decision: Not Verified.")
        else:
            st.write("User ID not found.")
    if verify_option == "Record For 5 Seconds":
        if user_id_exists(user_id):
            if st.button("Start Recording for Verification After Enrollment"):
                recording, samplerate = record_audio()
                save_wav('verify_after_enrollment_5_seconds.wav', recording, samplerate)
                new_embeddings = extract_embeddings('verify_after_enrollment_5_seconds.wav')
                similarity, is_same_speaker = verify_user(user_id, new_embeddings)
                number = similarity
                percentage = number * 1000
                formatted_percentage = f"{percentage:.2f}%"
                if percentage < 0:
                    st.write("Similarity score: 0%")
                elif percentage > 100.00:
                    st.write("Similarity score: 100%")
                else:
                    st.write(f"Similarity score: {formatted_percentage}")
                if similarity is not None:

                    if is_same_speaker:
                        st.write("Decision: Verified.")
                    else:
                        st.write("Decision: Not Verified.")
        else:
            st.write("User ID not found.")
    if verify_option == "Record Using Specific Duration":
        if user_id_exists(user_id):
            duration = st.slider("Select recording duration for verification (seconds)", 1, 10, 3,
                                 key="verify_after_enrollment_duration")
            if st.button("Start Recording for Verification After Enrollment"):
                recording, samplerate = record_audio_with_duration(duration)
                save_wav('verify_after_enrollment_5_seconds.wav', recording, samplerate)
                new_embeddings = extract_embeddings('verify_after_enrollment_5_seconds.wav')
                similarity, is_same_speaker = verify_user(user_id, new_embeddings)
                number = similarity
                percentage = number * 1000
                formatted_percentage = f"{percentage:.2f}%"
                if percentage < 0:
                    st.write("Similarity score: 0%")
                elif percentage > 100.00:
                    st.write("Similarity score: 100%")
                else:
                    st.write(f"Similarity score: {formatted_percentage}")
                if similarity is not None:

                    if is_same_speaker:
                        st.write("Decision: Verified.")
                    else:
                        st.write("Decision: Not Verified.")
        else:
            st.write("User ID not found.")
    if verify_option == "Upload Audio":
        verify_file = st.file_uploader("Choose a .wav file for verification after enrollment", type="wav", key="verify_after_enrollment_upload")
        if verify_file is not None:
            if user_id_exists(user_id):
                with open("verify_after_enrollment.wav", "wb") as f:
                    f.write(verify_file.getbuffer())
                st.success("File uploaded successfully")

                # Extract embeddings for verification after enrollment
                new_embeddings = extract_embeddings("verify_after_enrollment.wav")
                similarity, is_same_speaker = verify_user(user_id, new_embeddings)
                number = similarity
                percentage = number * 1000
                formatted_percentage = f"{percentage:.2f}%"
                if percentage < 0:
                    st.write("Similarity score: 0%")
                elif percentage > 100.00:
                    st.write("Similarity score: 100%")
                else:
                    st.write(f"Similarity score: {formatted_percentage}")
                if similarity is not None:
                    if is_same_speaker:
                        st.write("Decision: Verified.")
                    else:
                        st.write("Decision: Not Verified.")
        else:
            st.write("User ID not found.")
