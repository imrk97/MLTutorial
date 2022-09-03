import speech_recognition as sr
import pyttsx3


class text_speech:

    def __init__(self):
        self.engine = pyttsx3.init('sapi5')
        self.engine.setProperty('rate', 160)
        #self.engine.setProperty('volume')
        self.engine.setProperty('volume', 1)

    def say(self, msg):
    	self.engine.say(msg)
    	self.engine.runAndWait()

class speech_text:
    def __init__(self):
        self.text = ""
        self.t_s = text_speech()
        self.recognizer = sr.Recognizer()

    def listen(self):
        print('listening.....')
        with sr.Microphone() as source2:
            try:
                self.recognizer.adjust_for_ambient_noise(source2)
                print('audio_text part')
                audio2 = self.recognizer.listen(source2)
                
                self.text = self.recognizer.recognize_google(audio2)
                self.text = self.text.lower()
                
                print("The spoken line is:  "+ self.text)
            except:
                self.text = ""
                self.t_s.say("Sorry, I didn't get you.")
                    
                    
s_t = speech_text()
s_t.listen()
                    
