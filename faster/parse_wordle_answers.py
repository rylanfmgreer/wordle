# author: rylan greer
# March 20, 2022

import numpy  as np
import pandas as pd

def get_possible_answers():
    words = pd.read_csv('word_lists/wordle-answers-alphabetical.txt', header=None)
    return words.iloc[:, 0].to_numpy()

def get_possible_guesses():
    words = pd.read_csv('word_lists/wordle-allowed-guesses.txt', header=None)
    return words.iloc[:, 0].to_numpy()

if __name__ == '__main__':
    print( get_possible_guesses() )