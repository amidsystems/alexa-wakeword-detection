# this module handles text to speech
# Only tested on windows machine
import pyttsx3


class vini_voice:
  def __init__(self, voice = "female"):
    if voice == 'female':
      self.voice_index = 1 # index 1 for female voice
    else:
      self.voice_index = 0 # index 0 for male voice
    self.speech_engine = pyttsx3.init()
    self.speech_engine.setProperty('rate', 150)
    self.voices=self.speech_engine.getProperty('voices')
    print("Using {} voice, from 2 available voices".format(voice))
    
  def speak(self, text):
    self.speech_engine.setProperty('voice',self.voices[self.voice_index].id)
    self.speech_engine.say(text)
    self.speech_engine.runAndWait()



