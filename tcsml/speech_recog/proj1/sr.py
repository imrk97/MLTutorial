import speech_recognition as sr
r = sr.Recognizer()

with sr.AudioFile('325612_david.wav') as source:
        audio_text = r.listen(source)
        try:
            text = r.recognize_google(audio_text)
            print('Converting audio to text...')
            print(text)
            with open('a.txt', 'w') as f:
                f.write(text)
                f.close()

        except:
            print("Sorry.. can't hear you.")