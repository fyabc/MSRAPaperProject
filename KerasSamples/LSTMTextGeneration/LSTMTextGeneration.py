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

        if len(words) == 1:
            if words[0] == 'retrain':
                Config['retrain'] = True

        if len(words) == 2:
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


def randomMaxLen(text, maxLen=ParamConfig['maxLen']):
    startIndex = random.randint(0, len(text) - maxLen - 1)
    return text[startIndex:startIndex + maxLen]


def randomLine(text, maxLen=ParamConfig['maxLen']):
    index = random.randint(0, len(text) - 1)
    prevEOL = text.rfind('\n', None, index)
    nextEOL = text.find('\n', index, None)

    prevEOL = 0 if prevEOL == -1 else prevEOL
    nextEOL = len(text) if nextEOL == -1 else nextEOL

    result = text[prevEOL:nextEOL + 1]
    if len(result) > maxLen:
        result = result[:maxLen]

    return result


def main():
    # parsing command line parameters
    parseCommandLine()

    outputFile = getOutputFile()

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

    if not Config['retrain']:
        try:
            print('Loading exist model...', end='')
            model = loadModel()
        except Exception as e:
            print('an error occurred when loading the model: %s' % e)
            Config['retrain'] = True

    if Config['retrain']:
        # build the model: 2 stacked LSTM
        print('Building model...', end='')
        model = Sequential()
        model.add(LSTM(ParamConfig['batchSize'], input_shape=(maxLen, charNum)))
        model.add(Dense(charNum))
        model.add(Activation('softmax'))

    optimizer = RMSprop(lr=ParamConfig['learningRate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print('done')

    # train the model, output generated text before each iteration
    for i in range(1, ParamConfig['iterationNum']):
        # test
        for j in range(ParamConfig['sampleNum']):
            # sentenceInit = randomLine(text)
            sentenceInit = randomMaxLen(text)

            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print()
                print(file=outputFile)
                print('----- diversity:', diversity, end='')
                print('----- diversity:', diversity, end='', file=outputFile)

                generated = ''

                sentence = sentenceInit
                generated += sentence

                print('----- Generating with seed: "' + sentence + '"')
                print('----- Generating with seed: "' + sentence + '"', file=outputFile)
                sys.stdout.write(generated)
                outputFile.write(generated)

                for _ in range(ParamConfig['generateLen']):
                    x = np.zeros((1, maxLen, charNum))
                    for t, char in enumerate(sentence):
                        x[0, t, char2index[char]] = 1.0

                    predicts = model.predict(x, verbose=0)[0]

                    nextChar = index2char[sample(predicts, diversity)]

                    generated += nextChar
                    sentence += nextChar
                    if len(sentence) > maxLen:
                        sentence = sentence[1:]

                    try:
                        sys.stdout.write(nextChar)
                        outputFile.write(nextChar)
                    except UnicodeEncodeError:
                        sys.stdout.write('$')
                        outputFile.write('$')
                    sys.stdout.flush()

                print()
                print(file=outputFile)
                outputFile.flush()

        print('Dumping model...', end='')
        try:
            dumpModel(model)
            print('done')
        except Exception as e:
            print('an error occurred when dumping the model: %s' % e)

        print()
        print('-' * 50)
        print('Iteration:', i)
        model.fit(X, y, batch_size=ParamConfig['batchSize'], nb_epoch=ParamConfig['epochNum'])

    outputFile.close()


if __name__ == '__main__':
    main()
