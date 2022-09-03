import speech_recognition as sr




def is_mp3(source):
    return source.split('.')[-1].lower() == 'mp3'

def convert_mp3(src):
    from os import path
    from pydub import AudioSegment
    dst = src.split('.')[0] + '.wav'
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
    return dst

def main(ARGS):
    audio_file = ARGS.file
    if is_mp3(audio_file):
        audio_file = convert_mp3(audio_file)
    r = sr.Recognizer()

    print(audio_file)

    #print(type(ARGS.file))
    with sr.AudioFile('male.wav') as source:
        audio_text = r.listen(source)
        try:
            text = r.recognize_google(audio_text)
            print('Converting audio to text...')
            print(text)
            
            with open(source.filename_or_fileobject.split('.')[0] + '.txt', 'w') as f:
                f.write(text)
                f.close()

        except:
            print("Sorry.. can't hear you.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")
    parser.add_argument('-f', '--file', type=str, help="Read from .wav or .mp3 file")
    ARGS = parser.parse_args()
    main(ARGS)