from nltk import word_tokenize
from word2number import w2n
from deepspeech1 import transcribe_streaming
import pandas as pd


def generate_otp(file_path):
    try:
        text = open(file_path).read()
    except Exception as e:
        print(e)
    #print(text)
    return text

def start_index(text_tokens):
    ans = -99

    units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

    for i in range(len(text_tokens)):
        if text_tokens[i] in units:
            ans = i
            break
    return ans

def get_nums(text):
    dict_nums = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'zero': '0'
    }

    print(''.join(list(pd.Series(text).map(dict_nums))))

def digits_num():
    transcribe_streaming('904571_ravi.wav')
    text = generate_otp("a.txt")
    text_tokens = word_tokenize(text)
    start = start_index(text_tokens)
    otp_tokens = text_tokens[start:]
    get_nums(otp_tokens)


def word_num():
    transcribe_streaming('326123_Zira.wav')
    text = generate_otp("a.txt")
    text_tokens = word_tokenize(text)
    start = start_index(text_tokens)
    otp_tokens = text_tokens[start:]
    print(w2n.word_to_num(' '.join(otp_tokens)))

digits_num()