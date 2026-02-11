import speech_recognition as sr

def listen_from_mic():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Listening... Please speak")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {text}")
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand your speech."
    except sr.RequestError:
        return "Speech service is unavailable."
