import os
import whisper
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import tempfile
import pyaudio

# Set FFmpeg path explicitly
os.environ["PATH"] += r";C:\ffmpeg\bin"  # Update with the correct path for your FFmpeg installation

# Load the Whisper model for ASR (Automatic Speech Recognition)
whisper_model = whisper.load_model("base")

# Load the pre-trained sentiment analyzer (VADER)
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load Hugging Face pre-trained intent analysis model
intent_classifier = pipeline("zero-shot-classification")

# Intent label mapping (this can be customized based on your use case)
intent_labels = {
    0: "Complaint",
    1: "Product Inquiry",
    2: "Feedback",
    3: "Purchase Intent",
    # Add other labels as needed
}

# Function to perform sentiment analysis
def analyze_sentiment(text):
    sentiment = sentiment_analyzer.polarity_scores(text)
    if sentiment['positive'] > 0.60:
        sentiment_label = 'Positive'
    elif sentiment['negative'] > 0.60:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    return sentiment_label, sentiment

# Function to perform intent analysis
def analyze_intent(text):
    intent_result = intent_classifier(text, candidate_labels=["Complaint", "Product Inquiry", "Feedback", "Purchase Intent", "Other"])
    intent = intent_result['labels'][0]  # Most likely intent
    intent_score = intent_result['scores'][0]
    return intent, intent_score

# Function to process the audio input and return transcriptions
def analyze_audio(file_name):
    print("Transcribing audio...")
    transcription = whisper_model.transcribe(file_name)["text"]
    print(f"Transcription: {transcription}")
    
    # Sentiment Analysis
    sentiment_label, sentiment_values = analyze_sentiment(transcription)
    print(f"Sentiment Analysis: {sentiment_label}")
    print(f"Sentiment Scores: Positive: {sentiment_values['positive']*100:.2f}%, Neutral: {sentiment_values['neutral']*100:.2f}%, Negative: {sentiment_values['negative']*100:.2f}%")

    # Intent Analysis
    intent_label, intent_score = analyze_intent(transcription)
    print(f"Intent Analysis: {intent_label} with score {intent_score:.2f}")
    
    print("\n---\n")

# Set up speech recognition
def listen_and_analyze():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    with microphone as source:
        print("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_name = tmp_file.name + ".wav"
        with open(tmp_file_name, "wb") as f:
            f.write(audio.get_wav_data())
        print(f"Audio saved to {tmp_file_name}")
        analyze_audio(tmp_file_name)
        os.remove(tmp_file_name)  # Clean up the temporary file

if __name__ == "__main__":
    while True:
        listen_and_analyze()
