import sounddevice as sd
import numpy as np
import speech_recognition as sr
from scipy.io.wavfile import write
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pyttsx3
import os

def get_audio_input():
    """
    Captures audio using `sounddevice`, saves it temporarily, and converts it to text using `speech_recognition`.
    """
    print("Listening...")
    fs = 44100  # Sample rate
    duration = 5  # Duration of recording in seconds
    recognizer = sr.Recognizer()

    try:
        # Record audio
        print("Recording for 5 seconds...")
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
        sd.wait()  # Wait for recording to finish
        print("Recording complete.")

        # Save the recording temporarily
        temp_wav_path = "temp_audio.wav"
        write(temp_wav_path, fs, audio_data)

        # Use speech_recognition to process the WAV file
        with sr.AudioFile(temp_wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists("temp_audio.wav"):
            os.remove("temp_audio.wav")

def load_gpt2():
    """
    Loads the GPT-2 model and tokenizer on GPU if available, otherwise CPU.
    """
    print("Loading GPT-2 model...")
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "gpt2"  # GPT-2 model name
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the model explicitly and move it to the selected device (GPU or CPU)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    print("Model loaded successfully.")
    return model, tokenizer, device

def get_response(model, tokenizer, device, user_input):
    """
    Generates a response using GPT-2 for the given user input.
    """
    # Create a structured prompt
    prompt = (
        "You are an assistant that answers questions concisely.\n"
        "Q: What is your name?\n"
        "A: My name is GPT-2.\n"
        f"Q: {user_input}\n"
        "A:"
    )

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move input tensors to the correct device

    # Generate response with constraints to avoid unnecessary continuation
    outputs = model.generate(
        inputs.input_ids, 
        max_length=50,  # Limit the length of the output
        num_return_sequences=1, 
        no_repeat_ngram_size=2,  # Prevent repetition
        do_sample=False,  # Greedy decoding (no randomness)
        temperature=0.7,  # Less randomness for focused responses
        top_p=0.9,  # Use nucleus sampling for better output
        top_k=50  # Limit the sampling pool for better quality responses
    )

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated answer part (after the "A:")
    if "A:" in response:
        response = response.split("A:")[1].strip()
    
    return response

def speak_text(text):
    """
    Converts text to speech using pyttsx3 for offline TTS.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    # Load GPT-2 model
    model, tokenizer, device = load_gpt2()

    while True:
        # Step 1: Get Audio Input
        user_input = get_audio_input()
        if user_input:
            # Step 2: Generate Response
            print("Generating response...")
            response = get_response(model, tokenizer, device, user_input)
            print(f"GPT-2 says: {response}")

            # Step 3: Convert Response to Speech
            speak_text(response)
