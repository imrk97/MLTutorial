#!/usr/bin/env python
# coding: utf-8

# In[1]:


from deepspeech import Model
import numpy as np
import os
import wave
import json
import time
from IPython.display import Audio


# In[2]:


model_file_path = 'C:/Users/rohan/Downloads/deepspeech-0.9.3-models.pbmm'
lm_file_path = 'C:/Users/rohan/Downloads/deepspeech-0.9.3-models.scorer'
beam_width = 100
lm_alpha = 0.93
lm_beta = 1.18

model = Model(model_file_path)
model.enableExternalScorer(lm_file_path)


# In[3]:


model.setScorerAlphaBeta(lm_alpha, lm_beta)
model.setBeamWidth(beam_width)


# In[4]:


stream = model.createStream()


# In[5]:


def read_wav_file(filename):
    with wave.open(filename, 'rb') as w:
        rate = w.getframerate()
        frames = w.getnframes()
        buffer = w.readframes(frames)

    return buffer, rate


# In[6]:


#from IPython.display import clear_output

def transcribe_streaming(audio_file):
    buffer, rate = read_wav_file(audio_file)
    offset=0
    batch_size = 2048
    text=''
    print(audio_file)
    time.sleep(2)
    os.system('cls' if os.name == 'nt' else 'clear')
    try:
        while offset < len(buffer):
            
            end_offset=offset+batch_size
            chunk=buffer[offset:end_offset]
            data16 = np.frombuffer(chunk, dtype=np.int16)

            stream.feedAudioContent(data16)
            text=stream.intermediateDecode()
            time.sleep(0.1)
            os.system('cls' if os.name == 'nt' else 'clear')
            #clear_output(wait=True)
            #print(text, end=' ')
            print(text)
            offset=end_offset


   
    except AttributeError:
       pass

    else:
        try:
            with open('a.txt', 'w') as f:
                    f.write(text)
                    f.close()
        except:
            print("ERROR opening File.")
    return True


# In[ ]:
 
#print(os.name)
#transcribe_streaming('234561_heera.wav')


# In[ ]:


#transcribe_streaming('male.wav')


# In[ ]:




