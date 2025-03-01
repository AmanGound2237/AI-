import vosk
import sys
import queue
import sounddevice as sd
import json
import sqlite3
from transformers import pipeline
from TTS.api import TTS
from sentence_transformers import SentenceTransformer

# Initialize AI model
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

# Initialize Text-to-Speech (TTS)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Initialize Sentence Transformer for Memory
memory_model = SentenceTransformer("all-MiniLM-L6-v2")
conn = sqlite3.connect("ai_memory.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, user_input TEXT, response TEXT)""")
conn.commit()

def save_memory(user_input, response):
    """Stores conversation history in memory."""
    c.execute("INSERT INTO memory (user_input, response) VALUES (?, ?)", (user_input, response))
    conn.commit()

def retrieve_memory(user_input):
    """Retrieves the most relevant past conversation."""
    c.execute("SELECT response FROM memory ORDER BY id DESC LIMIT 5")
    past_responses = c.fetchall()
    return past_responses[-1][0] if past_responses else ""

def text_to_speech(text):
    """Converts text to speech and plays it."""
    tts.tts_to_file(text=text, file_path="response.wav")
    import os
    os.system("start response.wav" if sys.platform == "win32" else "aplay response.wav")

# Initialize Vosk Speech Recognition
model = vosk.Model("model")  # Ensure you have a Vosk model downloaded
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def recognize_speech():
    """Captures and recognizes speech using Vosk."""
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
        recognizer = vosk.KaldiRecognizer(model, 16000)
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                return result.get("text", "")

# Main AI Loop
while True:
    print("Listening...")
    user_input = recognize_speech()
    print("You:", user_input)
    
    if user_input.lower() in ["exit", "quit", "stop"]:
        print("Goodbye!")
        break
    
    memory_response = retrieve_memory(user_input)
    response = chatbot(user_input, max_length=100, do_sample=True)
    ai_text = response[0]['generated_text']
    final_response = memory_response if memory_response else ai_text
    print("AI:", final_response)
    save_memory(user_input, final_response)
    text_to_speech(final_response)
