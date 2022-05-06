# author: rylan greer
# March 20, 2022

import numpy as np
from parse_wordle_answers import *
from wordle_info_classes import *

import pandas as pd
def turn_slice_into_output(m):
    array = pd.DataFrame(m)
    array.index =  ['Letter can fit in position {i}'.format(i=i)
            for i in range(1, 6)] + ['Yellow tile for letter', 'Black tile for letter']
    array.columns = [chr(97 + i) for i in range(26)]
    return array

if __name__ == '__main__':
    print('______')
    print('Testing with standard Wordle dictionary')


    guesses = get_possible_guesses()[:]
    answers = get_possible_answers()[:]

    info   = Info(5)
    D      = WordleDic(np.array(answers))
    killed = D.make_all_guesses(info)

    guesses = get_possible_guesses()
    killed         = pd.DataFrame( killed )
    killed.index   = guesses
    killed.columns = guesses

    killed_mean = killed.mean(axis=1)
    bw = killed_mean[ killed_mean == killed_mean.max() ].index[0]

    print(' Best word:', bw)
    