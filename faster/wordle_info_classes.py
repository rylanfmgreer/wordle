# author: rylan greer
# march 20, 2022

import numpy as np
import pandas as pd # for debugging
from typing import Union
from datetime import datetime
from parse_wordle_answers import *
from numba_helpers import num_killed

class Info:
    def __init__(self, n: int):
        """
            Initialize an information object.
            An information object can be thought of as the sum of all information
            displayed in the "grid" in wordle -- the green, yellow, and black tiles, 
            in conjunction with the letters within those tiles.

            Infomation is stored as a (n + 2, 26) boolean array.

            The first n rows represent whether or not a letter can be present
            in a specific place in the word.

            The second-last row represents whether the letter is present at all

            The last row represents whether the letter is excluded.
        """

        self.array = np.ones( (n + 2, 26)).astype(bool)
        self.array[-2:, :] = False
        self.shape = self.array.shape

        self.string_word = ''


    def update_info(self, guess_word, true_word, inplace=False):
        new_info = update_info(guess_word, true_word, self)
        if inplace:
            self = new_info
        return new_info


    def __str__(self) -> str:
        return self.array.__str__()


    def __len__(self) -> int:
        return self.shape[0] -2


    def __repr__(self) -> str:
        return self.__str__()


    def __getitem__(self, tup):
        return self.array[tup[0], tup[1]]


    def compatible(self, word ) -> bool:
        return compatible(word, self)


    def copy(self):
        """
            Make a deep copy of our information
        """
        n              = self.shape[0] - 2
        new_word       = Info(n)
        new_word.array = self.array.copy()
        return new_word


    def compatible(self, word):
        return compatible(word, self)


    def score_next_best_word(self, potential_guess_words,
                             potential_solutions, verbose=False):
        """
            Given a list of potential words, determine which one is the best guess
            This returns an array with the score of each word -- higher score corresponds
            to better word.

        """
        scores = np.empty( len(potential_guess_words) ).astype(int)
        solutions_dic = WordleDic(potential_solutions)
        best_score_so_far = 0
        best_word_so_far  = ''

        for i in range( len(potential_guess_words) ):
            guess_word = potential_guess_words[i]
            num_killed = 0

            for j in range( len(potential_solutions) ):
                potential_true_word = potential_solutions[j]
                new_info = update_info(guess_word, potential_true_word, self)
                killed = solutions_dic.find_compatibility(new_info)
                num_killed += killed.sum()

            scores[i] = num_killed
            if num_killed > best_score_so_far:
                best_word_so_far  = guess_word
                best_score_so_far = num_killed
            if verbose:
                print('Word: {w1}. Best word so far: {w2}'.format(w1=guess_word, w2=best_word_so_far))

        return scores


    # private:
    def _letter_to_index(self, ch: str):
        return ord(ch.lower()) - 97


class WordleDic:
    """
        A dictionary containing all the words' representations as info.
        This is used to apply vectorized calculations to speed up the execution
    """

    def __init__(self, words):

        n = len(words)
        assert n > 0
        word_len = len( words[0] )

        # words can be a list of strings or a list of their info classes
        for i in range(n):
            assert len(words[i]) == word_len
            if isinstance(words[i], str):
                words[i] = info_from_word(words[i])

        # copy the words into an array for the whole dict
        self.array = np.zeros((n, words[0].shape[0], words[0].shape[1]))
        for i in range(n):
            self.array[i] = words[i].array.copy()


    def find_compatibility(self, info):
        """
            Given an instance of Info, determine which words in the dictionary are compatible.

            algorithm:
            1. if we have a False in a location in Info corresponding to
               a letter in word, then it is not compatible
            2. If the info says a letter must be present, and it is not present in
               the word, then it is not compatible
            3. If the info says that a letter must NOT be present, and it is present
               in the word, then the word is not compatible
            4. If none of the triggers 1-3 are hit, then the word is compatible
        """

        assert self.array[0].shape == info.shape
        n = self.array[0].shape[0] - 2

        # Check 1 in algorithm:
        word_arr    = self.array[:, :n, :]
        info_arr    = info.array[:n, :]
        violation_1 = np.logical_and( np.logical_not(info_arr), word_arr ).any(axis=1).any(axis=1)

        # Checks 2 and 3 in algorithm:
        word_arr  = self.array[:, -2:]
        info_arr  = info.array[-2:]
        violation_2 = np.logical_and( info_arr, np.logical_not(word_arr) ).any(axis=1).any(axis=1)

        return np.logical_or(violation_1, violation_2)


    def make_all_guesses(self, info, verbose=True):
        """
            Vectorized version of making a guess and revealing info.
            algorithm:
            1. (Green tiles):   If guess_word and true_word have a match, then
               we set all elements in the corresponding index in info to False,
               and set the index of the corresponding letter to True. This means that,
               for the position on the grid, the letter in question may be present, and 
               all other letters must not be present.

            2. (Yellow tiles):   If guess_word and true_word have letters in common,
               then we update the second-last row in info such that we say that the
               letters must be present.

            3. (Black tiles I):  If a letter is present in guess_word but not in true_word,
               then we update the final row to say that that letter must NOT be present

            4. (Black tiles II): If a letter is present in guess_word but not in true_word,
               update the first n rows (the positional rows) to exclude that letter.
        """

        n        = info.shape[0] - 2   # number of letters in the word
        n_words  = self.array.shape[0] # number of words in the dictionary
        start, stop = datetime.now(), datetime.now()

        # new_info shape: n_words x n_words x (word_length + 2) x 26
        new_info = _array_repeat(info.array.copy(), n_words, n_words)
        
        # 1. Check for letters that match in location
        # arr shape:    n_words x (word_length) x (26)
        # green shape:  n_words x n_words x (word_length) x 26
        if verbose:
            print('\nProcess begins at ', start)
            print('\nCalculating greens and updating...')
        guess_arr, true_arr = self.array[:, :n], self.array[:, :n]
        green               = np.einsum('abc,dbc->adbc', guess_arr, true_arr, casting='no')
        match_found         = green.any(axis=3, keepdims=True)
        new_info[:, :, :n]  = np.where(match_found, green, new_info[:, :, :n])

        # 2. Check to see if any letters exist in both:
        # arr shape:    n_words x 1 x 26
        # yellow shape: n_words x n_words x 1 x 26
        if verbose:
            stop = datetime.now()
            print('Time taken:', stop-start)
            start = stop
            print('\nCalculating yellows and updating...')

        guess_arr, true_arr = self.array[:, n], self.array[:, n]
        yellow              = np.einsum('ac,bc->abc', guess_arr, true_arr, casting='no')
        new_info[:, :, n]   = np.where(yellow, yellow, new_info[:, :, n])

        # 3. Check to see which letters are in guess but not in true:
        # arr shape:    n_words x 1 x 26
        # black shape:  n_words x n_words x 1 x 26
        if verbose:
            stop = datetime.now()
            print('Time taken:', stop-start)
            start = stop
            print('\nCalculating blacks and updating...')
        guess_arr, true_arr     = self.array[:, n],  self.array[:, (n + 1)]
        black                   = np.einsum('ac,bc->abc', guess_arr, true_arr)
        new_info[:, :, (n + 1)] = np.where(black, black, new_info[:, :, (n + 1)])

        # 4. If we have a black in new_info, ensure that the columns where
        # it exists are marked to false!
        new_info[:, :, :n] = np.where(new_info[:, :, (n + 1):(n + 2)], False, new_info[:, :, :n])
        
        if verbose:
            stop = datetime.now()
            print('Time taken:', stop - start)
            start = stop

        if verbose:
            print('\nCalculating which words are knocked out...')
        word_arr = self.array
        killed = num_killed(new_info, word_arr)


        if verbose:
            stop = datetime.now()
            print('Time taken:', stop - start)
            print('Process complete at ', stop)

        return np.array(killed)


def info_from_word(word: str) -> Info:
    """
            Initialize an Info object from a known word where we know
            exactly what the word is
    """
    info = Info( len(word) )
    info.string_word = word
    n    = len(word)
    info.array = np.zeros( (n + 2, 26) ).astype(bool)

    for i in range(n):
        ch  = word[i]
        idx = info._letter_to_index(ch)
        info.array[i, idx] = True

    # second last: does the letter exist in the word
    info.array[n, :] = info.array[:n].any(axis=0)

    # last: is the letter KNOWN NOT PRESENT in the word
    info.array[n + 1, :] = np.logical_not(info.array[n, :])

    return info


def info_from_dict(greens, yellows, blacks) -> Info:
    """
        Create an Info object
    """
    # TODO: implement this function
    pass


def compatible(word: Union[Info, str], info: Info) -> bool:
    """

        Given the information in info, is 'word' a possibility?
        
        algorithm:
            1. if we have a False in a location in Info corresponding to
               a letter in word, then it is not compatible
            2. If the info says a letter must be present, and it is not present in
               the word, then it is not compatible
            3. If the info says that a letter must NOT be present, and it is present
               in the word, then the word is not compatible
            4. If none of the triggers 1-3 are hit, then the word is compatible

    """

    # allow for various types of input for word
    if isinstance(word, str):
        word = info_from_word(word)
    
    # sanity check
    assert word.shape == info.shape

    # Check 1 in algorithm:
    n         = word.shape[0] - 2
    word_arr  = word.array[:n, :]
    info_arr  = info.array[:n, :]
    violation = np.logical_and( np.logical_not(info_arr), word_arr ).any()
    if violation:
        return False

    # Checks 2 and 3 in algorithm:
    word_arr  = word.array[-2:]
    info_arr  = info.array[-2:]
    violation = np.logical_and( info_arr, np.logical_not(word_arr) ).any()
    if violation:
        return False

    # if we reach this, no violation has been seen and so the word is OK
    return True

def update_info(guess_word, true_word, info_):
    """

        When we make a guess in wordle, we get new information.
        Use this information to update the new info you have.

        algorithm:
            1. (Green tiles):   If guess_word and true_word have a match, then
               we set all elements in the corresponding index in info to False,
               and set the index of the corresponding letter to True. This means that,
               for the position on the grid, the letter in question may be present, and 
               all other letters must not be present.

            2. (Yellow tiles):   If guess_word and true_word have letters in common,
               then we update the second-last row in info such that we say that the
               letters must be present.

            3. (Black tiles I):  If a letter is present in guess_word but not in true_word,
               then we update the final row to say that that letter must NOT be present

            4. (Black tiles II): If a letter is present in guess_word but not in true_word,
               update the first n rows (the positional rows) to exclude that letter.

    """
    if isinstance(guess_word, str):
        guess_word = info_from_word(guess_word)
    if isinstance(true_word, str):
        true_word = info_from_word(true_word)

    assert len(guess_word) == len(true_word)
    assert len(true_word) == len(info_)

    info = info_.copy()
    n    = len(info)

    # Part 1 (Green tiles)
    guess_arr = guess_word.array[:n]
    true_arr  = true_word.array[:n]
    match     = np.logical_and(guess_arr, true_arr)
    matches   = match.any(axis=1)
    location  = match.argmax(axis=1)
    for i in range(n): # len( matches )
        obj = matches[i]
        if obj:
            idx = location[i]
            info.array[i, :]   = False
            info.array[i, idx] = True

    # Part 2 (Yellow tiles)
    guess_arr     = guess_word.array[n]
    true_arr      = true_word.array[n]
    update        = np.logical_and(guess_arr, true_arr)
    info.array[n] = np.logical_or(info.array[n], update)

    # Part 3 (Black tiles I):
    guess_arr         = guess_word.array[n]
    true_arr          = true_word.array[n + 1]
    update            = np.logical_and(guess_arr, true_arr)
    info.array[n + 1] = np.logical_or(info.array[n + 1], update)

    # Part 4 (Black Tiles II)
    # there are some redundant checks
    update = info.array[n + 1]
    for i in range( len(update) ):
        if update[i]:
            info.array[:n, i] = False

    # updates are complete
    return info       


def _array_repeat(arr, n_axis0=0, n_axis1=0):
    new_arr = np.expand_dims(arr, axis=[0, 1])
    new_arr = np.tile(arr, [n_axis0, n_axis1] + [1] * len(arr.shape))
    return new_arr


if __name__ == '__main__':
    A = np.array([True, False])
    B = np.array([False, True])
    C = np.einsum('i,j->ij', A, B)
    print(C)