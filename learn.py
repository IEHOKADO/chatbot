from keras.layers import Dense, LSTM,Embedding, Bidirectional, Activation
from keras.models import Model, Sequential, load_model
from keras.callbacks import LambdaCallback
from keras.optimizers import RMSprop
import tinysegmenter
import numpy as np
import json
import sys
import os
from config import *

EPOCHS = 15  #エポック数
BATCH_SIZE = 64  #バッチサイズ
maxlen = 15  #入力単語数の限界

def get_dict():
    with open(PATH2DICT[0], encoding = 'utf-8') as f:
        char2idx = json.load(f)
    with open(PATH2DICT[1], encoding = 'utf-8') as f:
        idx2char = json.load(f)
    return char2idx, idx2char

def vectorization(char2idx):
    x = []
    y = []
    names = sorted(os.listdir(PATH2QUESTION))
    print(names)
    for i, name in enumerate(names):
        with open(PATH2QUESTION + '/' + name, encoding = 'utf-8') as f:
            questions = f.read().lower().splitlines()
        for question in questions:
            chars = tinysegmenter.tokenize(question)  #分かち書き
            x.append([char2idx[char] for char in chars] + [0] * (maxlen - len(chars)))
            y.append(i + 1)
    return np.array(x), np.array(y)

def get_model(m, n):
    model = Sequential()
    model.add(Embedding(m + 1, BATCH_SIZE))
    model.add(Bidirectional(LSTM(BATCH_SIZE)))
    model.add(Dense(n))
    model.add(Activation("softmax"))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    model.summary()
    return model

def chat(sentence, model, char2idx, idx2char):
    chars = tinysegmenter.tokenize(sentence)  #分かち書き
    print(chars)
    x = []
    for char in chars:
        try:
            x.append(char2idx[char])
        except:
            x.append(0)
    x = [x + [0] * (maxlen - len(x))]
    print(x)
    probas = model.predict_proba(np.array(x))  #答えの確率を取得
    print(probas)
    max_proba = np.amax(probas)
    if max_proba > 0.7:
        idx = np.argmax(probas)  #最大の値を取得
    else:
        idx = 0  #50%以上の確率でなければ答えない
    print(idx)
    answer = idx2char[str(idx)]  #答えを辞書から取得
    return answer

def main():
    args = sys.argv  #コマンドライン引数を取得
    char2idx, idx2char = get_dict()
    m = len(char2idx)  #単語の総数
    n = len(idx2char)  #答えの数
    x, y = vectorization(char2idx)  #x,yベクトル
    model = get_model(m, n)  #モデルの生成
    if '--train' in args:
        model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE)  #繰り返し訓練させる
        model.save(filepath=PATH2MODEL)  #モデルの保存
    else:
        model = load_model(PATH2MODEL)  #学習済みモデルを使用する
    print('===== S T A R T =====')
    while (True):
        sentence = input().lower()
        print(chat(sentence, model, char2idx, idx2char))
    
if __name__ == '__main__':
    main()