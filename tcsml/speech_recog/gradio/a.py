from transformers import pipeline

p = pipeline("automatic-speech-recognition")



import gradio as gr

'''def transcribe(audio):
    text = p(audio)["text"]
    return text

gr.Interface(
    fn=transcribe, 
    inputs=gr.inputs.Audio(source="microphone", type="filepath"), 
    outputs="text",
    live=True).launch()'''

import gradio as gr

def transcribe(audio, state=""):
    text = p(audio)["text"]
    state += text + " "
    return state, state

gr.Interface(
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath"), 
        "state"
    ],
    outputs=[
        "textbox",
        "state"
    ],
    live=True).launch()