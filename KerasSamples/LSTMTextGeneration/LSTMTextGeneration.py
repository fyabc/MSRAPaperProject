#! /usr/bin/python3
# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals

import sys
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop

from config import Config, ParamConfig
from preprocess import *

__author__ = 'fyabc'


def parseCommandLine():
    argc = len(sys.argv)
    for i in range(1, argc):
        words = sys.argv[i].split(':')

        if len(words) != 2:
            continue

        if words[0] == 'data':
            Config['dataFiles'] = words[1]


def sample(predicts, temperature=ParamConfig['temperature']):
    # helper function to sample an index from a probability array
    predicts = np.asarray(predicts).astype('float64')
    predicts = np.log(predicts) / temperature
    exp_predicts = np.exp(predicts)
    predicts = exp_predicts / np.sum(exp_predicts)
    probabilities = np.random.multinomial(1, predicts, 1)
    return np.argmax(probabilities)


def main():
    # parsing command line parameters
    parseCommandLine()

    maxLen = ParamConfig['maxLen']

    text = readFile()
    chars = getChars(text)
    char2index, index2char = getCharMap(chars)
    sentences, nextChars = getSentences(text, maxLen)

    charNum = len(chars)

    print('Corpus length:', len(text))
    print('Total chars:', charNum)
    print('Number sequences:', len(sentences))

    print('Vectorization...', end='')
    X, y = sentence2vector(sentences, nextChars, chars, char2index)
    print('done')

    # build the model: 2 stacked LSTM
    print('Building model...', end='')

    model = Sequential()
    model.add(LSTM(ParamConfig['batchSize'], input_shape=(maxLen, charNum)))
    model.add(Dense(charNum))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=ParamConfig['learningRate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print('done')

    print('Dumping model...', end='')

    dumpModel(model)

    print('done')

    # train the model, output generated text after each iteration
    for i in range(1, ParamConfig['iterationNum']):
        print()
        print('-' * 50)
        print('Iteration:', i)
        model.fit(X, y, batch_size=ParamConfig['batchSize'], nb_epoch=ParamConfig['epochNum'])

        startIndex = random.randint(0, len(text) - maxLen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[startIndex: startIndex + maxLen]
            generated += sentence

            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for _ in range(400):
                x = np.zeros((1, maxLen, charNum))
                for t, char in enumerate(sentence):
                    x[0, t, char2index[char]] = 1.0

                predicts = model.predict(x, verbose=0)[0]

                nextChar = index2char[sample(predicts, diversity)]

                generated += nextChar
                sentence = sentence[1:] + nextChar

                sys.stdout.write(nextChar)
                sys.stdout.flush()

            print()


if __name__ == '__main__':
    main()
