from google import genai
from google.genai import types
import pyttsx3
import speech_recognition as sr

# Set up your Gemini API key
GEMINI_API_KEY = "API-KEY"

# Initialize the Google Gen AI client
client = genai.Client(api_key=GEMINI_API_KEY)

def listen_to_user():
    """Uses microphone to capture speech and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your question...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

def get_ai_response(prompt):
    """Gets response from the Gemini 1.5 model."""
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        answer = response.text
        print(f"AI Response: {answer}")
        return answer
    except Exception as e:
        print(f"Error fetching AI response: {e}")
        return "I'm sorry, I couldn't process that."

def speak_response(response):
    """Converts text to speech and plays it back."""
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

if __name__ == "__main__":
    print("Welcome to the AI Voice Chatbot!")
    print("Please ask your question after the prompt.")
    while True:
        user_input = listen_to_user()
        if user_input is None:
            print("Let's try again.")
            continue

        print(f"Your Question: {user_input}")

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Goodbye!")
            break

        ai_response = get_ai_response(user_input)
        print(f"AI's Response: {ai_response}")
        speak_response(ai_response)

