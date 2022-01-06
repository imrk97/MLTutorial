# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 10:00:40 2021

@author: rohan
"""

import nltk #we'll use nltk.pos_tag() for pos tagging 
#for importing speeches
from nltk.corpus import state_union
#for custom tokenizing 
from nltk.tokenize import PunktSentenceTokenizer

#importint train text
train_text = state_union.raw("2005-GWBush.txt")
#target text for pos tag
bush_speech = state_union.raw("2006-GWBush.txt")


#making custom tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
#tokens from custom tokenizer
bush_speech_tokens = custom_sent_tokenizer.tokenize(bush_speech)



def process_content():
    try:
        for i in bush_speech_tokens:
            words = nltk.word_tokenize(i)
            tagged_pos = nltk.pos_tag(words)
            '''chunkGram = r"""Chunk: {<RB.?>*}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged_pos)'''
            chunkGram = r"""Chunk: {<.*>+}
                            }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged_pos)
            print(chunked)
            chunked.draw()
    except Exception as e:
        print(str(e))
process_content()
