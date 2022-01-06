# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 16:38:05 2021

@author: rohan
"""

def max_occur(a):
    
    s = a.lower()
    char_dict = {}
    for char in s:
        if not char_dict.__contains__(char):
            char_dict[char] = 1
        else:
            char_dict[char] = char_dict[char] + 1
    return max(char_dict.values())


print(max_occur("Rohanr"))