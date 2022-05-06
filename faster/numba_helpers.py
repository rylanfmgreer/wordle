# rylan greer
# March 23, 2022

import numpy  as np
import pandas as pd
from numba import jit

"""
    This is a set of functions using Numba which
    provide a relatively good run time for what I believe
    is ultimately an O(n^3) process.

    Trying to implement these calculations naively using
    tensordots in Numpy would require at least (n_words)^3 x 26 x 7 bytes.
    For 2300 words this is over 200GB.

    The memory efficiency of these is pretty good in comparison
"""

@jit
def _green_compare(info_array, word_array, max_letters=26):
    """
        info_array: (n_letter + 2) x 26 array
        word_array: (n_letter + 2) x 26 array

        algorithm:
            for every row, check to see if:
                -the info says we CAN'T have the letter in that position
                -the word DOES have the letter in that position
            if this is the case for any letter, return True (the word has been killed)
            if this is not the case for any letter, return False (the word has not been killed)
    """
    n_letter = info_array.shape[0] - 2
    for r in range( n_letter ):
        for c in range( max_letters ):
            if not( info_array[r, c] ) and word_array[r, c]:
                return True
    return False


@jit
def _yellow_compare(info_array, word_array, max_letters=26):
    """
        info_array: (n_letter + 2) x 26 array
        word_array: (n_letter + 2) x 26 array

        algorithm:
            see if the Yellow row says the letter exists, but the word says the
            letter is not present. If there is a letter which is present in the yellow row
            of Info, but not present in the yellow row of the word, return True (the word has been killed).
            Otherwise, return False (the wod has not been killed)
    """
    info_compare = info_array[-2]
    word_compare = word_array[-1]
    for r in range( max_letters ):
        if info_compare[r] and word_compare[r]:
            return True
    return False


@jit
def _black_compare(info_array, word_array, max_letters=26):
    """
        info_array: (n_letter + 2) x 26 array
        word_array: (n_letter + 2) x 26 array

        algorithm:

    """
    info_compare = info_array[-1]
    word_compare = word_array[-2]
    for r in range( max_letters ):
        if info_compare[r] and word_compare[r]:
            return True
    return False


@jit
def _killed(info_array, word_array, max_letters=26):
    """
        info_array: (n_letter + 2) x 26 array
        word_array: (n_letter + 2) x 26 array

        it is not immediately clear which ordering will be fastest
        but my gut tells me that we are most likely to kill
        using the black tiles, then the yellow tiles, then the greens.
    """
    if _black_compare(info_array, word_array, max_letters=max_letters):
        return True
    if _yellow_compare(info_array, word_array, max_letters=max_letters):
        return True
    if _green_compare(info_array, word_array, max_letters=max_letters):
        return True

    return False


@jit
def num_killed(all_infos, all_words):
    '''
        info_array: n_words x n_words x (n_letters + 2) x 26 array
        all_words:  n_words x (n_letters + 2) x 26 array
    '''

    # set up shapes and output
    n_words     = all_infos.shape[0]
    max_letters = all_infos.shape[-1]

    ai_nrow = all_infos.shape[0]
    ai_ncol = all_infos.shape[1]
    aw_nrow = all_words.shape[0]

    output      = [ [0] * ai_ncol for i in range(n_words) ]


    for ai_r in range(ai_nrow):
        for ai_c in range(ai_ncol):
            for aw_r in range(aw_nrow):
                info_arr = all_infos[ai_r, ai_c]
                word_arr = all_words[aw_r]
                k = _killed(info_arr, word_arr, max_letters=max_letters)
                if k:
                    output[ai_r][ai_c] += 1

    return output 
