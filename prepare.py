import tinysegmenter
import json
import os
from config import *

def get_text():
    text = []
    questions = sorted(os.listdir(PATH2QUESTION))
    for question in questions:
        with open(PATH2QUESTION + '/' + question, encoding='utf-8') as f:
            text += f.read().lower().splitlines()
    return text

def get_answers():
    with open(PATH2ANSWER, encoding='utf-8') as f:
        answers = f.read().splitlines()
    return answers

def make_dictionary(text, answers):
    chunk = []
    for d in text:
        chunk += tinysegmenter.tokenize(d)  #分かち書きする
    vocab = sorted(set(chunk))  #辞書順にソート(同じ単語は消える)
    char2idx = dict((char, i + 1) for i, char in enumerate(vocab))
    idx2char = dict((i + 1, answer) for i, answer in enumerate(answers))
    idx2char[0] = 'わからないよーーー！'
    return char2idx, idx2char

def save_dictionary(char2idx, idx2char):
    with open(PATH2DICT[0], 'w', encoding = 'utf-8') as f:
        json.dump(char2idx, f, indent = 4, ensure_ascii = False)
    with open(PATH2DICT[1], 'w', encoding = 'utf-8') as f:
        json.dump(idx2char, f, indent = 4, ensure_ascii = False)

def main():
    text = get_text()
    answers = get_answers()
    char2idx, idx2char = make_dictionary(text, answers)
    save_dictionary(char2idx, idx2char)

if __name__ == '__main__':
    main()
