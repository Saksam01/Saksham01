from google import genai
from google.genai import types
import pyttsx3

# Set up your Gemini API key
GEMINI_API_KEY = "API-KEY" 

# Initialize the Google Gen AI client
client = genai.Client(api_key=GEMINI_API_KEY)

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
    print("Welcome to the AI Text Prompt Chatbot with Prompt Engineering!")
    print("Type your prompt below and press Enter to get a response.")
    while True:
        user_input = input("Your Prompt: ")

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Goodbye!")
            break

        print("Fetching response...\n")

        # Get response from Gemini API
        ai_response = get_ai_response(user_input)

        # Speak the response
        speak_response(ai_response)
