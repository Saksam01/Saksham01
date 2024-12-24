import json
import pyttsx3
import speech_recognition as sr
from llamaapi import LlamaAPI

def speak(engine, text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def get_user_input(recognizer, microphone):
    """Capture user input from the microphone."""
    try:
        print("Listening...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        print("Processing...")
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

def get_llama_response(api_key, user_input):
    """Send the user's input to the LlamaAPI and get a response."""
    llama = LlamaAPI(api_key)
    api_request_json = {
        "model": "llama3.1-70b",  # Replace with the desired model
        "messages": [
            {"role": "user", "content": user_input},
        ],
        "stream": False,
    }
    try:
        response = llama.run(api_request_json)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
    except Exception as e:
        return f"Error while communicating with LlamaAPI: {e}"

def main():
    """Main function to run the voice chatbot."""
    # Replace <your_api_token> with your actual API key
    api_key = "API-KEY"

    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("AI Voice Chatbot is ready. Say 'exit' to quit.")
    speak(engine, "Hello! I am your AI assistant. How can I help you today?")

    while True:
        user_input = get_user_input(recognizer, microphone)
        print(f"You: {user_input}")

        if user_input.lower() == "exit":
            speak(engine, "Goodbye!")
            print("Goodbye!")
            break

        response = get_llama_response(api_key, user_input)
        print(f"Llama: {response}")
        speak(engine, response)

if __name__ == "__main__":
    main()
