import speech_recognition as sr
import pyttsx3
import pandas as pd
import wave
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listens to the user's voice input and converts it to text."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text, audio
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None, None
    except sr.RequestError:
        print("There seems to be an issue with the recognition service.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def analyze_sentiment(text):
    """Analyzes sentiment of the given text."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)

    positive = sentiment_score['pos'] * 100
    negative = sentiment_score['neg'] * 100
    neutral = sentiment_score['neu'] * 100

    if sentiment_score['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_score['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, positive, negative, neutral

def analyze_tone(audio):
    """Analyzes the tone of the user's audio input."""
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())

        with wave.open("temp_audio.wav", "rb") as wav_file:
            frames = wav_file.readframes(-1)
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            signal = np.frombuffer(frames, dtype=np.int16)

            avg_amplitude = np.mean(np.abs(signal))
            energy = np.sum(signal ** 2) / len(signal)

            if avg_amplitude > 5000 and energy > 0.01 * (2 ** (8 * sample_width)):
                return "Excited"
            elif avg_amplitude < 2000 and energy < 0.005 * (2 ** (8 * sample_width)):
                return "Calm"
            else:
                return "Neutral"
    except Exception as e:
        print(f"An error occurred during tone analysis: {e}")
        return "Neutral"

def analyze_intent_with_tone(text, tone):
    """Analyzes the user's intent based on text and tone."""
    text = text.lower()
    intents = {
        "greeting": ["hi", "hello", "hey", "good morning", "good evening", "greetings"],
        "complaint": ["frustrated", "angry", "disappointed", "complain", "issue", "bad service", "problem"],
        "praise": ["great", "awesome", "amazing", "best", "excellent"],
        "question": ["what", "how", "why", "where", "who"],
        "exit": ["exit", "bye", "goodbye", "quit", "see you"]
    }

    for intent, keywords in intents.items():
        if any(keyword in text for keyword in keywords):
            if tone == "Excited" and intent == "greeting":
                return "enthusiastic_greeting"
            elif tone == "Calm" and intent == "complaint":
                return "polite_complaint"
            return intent
    return "unknown"

def provide_solution(sentiment, intent):
    """Provides a solution or response based on sentiment and intent."""
    responses = {
        "enthusiastic_greeting": "Hello! You sound excited. How can I assist you today?",
        "polite_complaint": "Thank you for calmly addressing your concern. Could you share more details?",
        "greeting": "Hello! How can I assist you today?",
        "complaint": "I'm sorry to hear you're upset. Could you share more details about the issue?",
        "praise": "Thank you for your kind words! Could you tell me what you liked the most?",
        "question": "Let me try to help you with your query. Could you provide more specifics?",
        "exit": "Goodbye! Have a great day!"
    }

    if intent in responses:
        return responses[intent]

    if sentiment == "Negative":
        return "I noticed some negative feedback. Could you clarify or share more details about your concern?"
    elif sentiment == "Positive":
        return "That's great! Could you share more details about what made you happy?"
    else:
        return "I'm not sure how to help with that. Could you provide more information?"

def display_table(sentiment, positive, negative, neutral, intent, solution, tone):
    """Displays the analysis results in a tabular format."""
    data = {
        "Sentiment": [sentiment],
        "Positive %": [f"{positive:.2f}"],
        "Negative %": [f"{negative:.2f}"],
        "Neutral %": [f"{neutral:.2f}"],
        "Intent": [intent],
        "Tone": [tone],
        "Solution": [solution]
    }
    df = pd.DataFrame(data)
    print(df)

def run_assistant():
    """Main function to run the voice assistant."""
    print("Voice Assistant is now running. Say 'exit' to stop.")
    while True:
        user_input, audio = listen()
        if user_input:
            if "exit" in user_input.lower():
                speak("Thank you for using the assistant. Goodbye!")
                break

            sentiment, positive, negative, neutral = analyze_sentiment(user_input)
            tone = analyze_tone(audio) if audio else "Neutral"
            intent = analyze_intent_with_tone(user_input, tone)
            solution = provide_solution(sentiment, intent)

            display_table(sentiment, positive, negative, neutral, intent, solution, tone)

            speak(f"Sentiment: {sentiment}. Positive: {positive:.2f}%. Negative: {negative:.2f}%. Neutral: {neutral:.2f}%. Tone: {tone}. Intent: {intent}.")
            speak(solution)
        else:
            speak("I didn't catch that. Could you please repeat?")

if __name__ == "__main__":
    run_assistant()

