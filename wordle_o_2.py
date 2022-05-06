# -*- coding: utf-8 -*-
#file: wordle_o.py
#author: rylan greer

## 1. Get five letter words

# dependency: 
import numpy  as np
import pandas as pd
import english_words

word_list = english_words.english_words_lower_alpha_set
word_length = 5
flw = set([word for word in word_list if 
           (len(word) == word_length) &
           (word[0] == word[0].lower()) for obj in word_list])
flw.remove('u.s.a')
n_words = len(flw)

"""## Vectorized calculation: functions"""

l2n = {}
alphabet = 'abcdefghijklmnopqrstuvwxyz'
for i in range(len(alphabet)):
  l2n[alphabet[i]] = i

flw_list = list(flw)

# Object oriented

class Word:
  def __init__(self, s):
    
    self.word = s
    self.grid = np.zeros( (1, len(s), len(alphabet)) ).astype(bool)
    for i in range( len(s) ):
      self.grid[0, i, l2n[ s[i] ]] = True

class Dictionary:
  def __init__(self, list_of_words=[]):
    self.words = np.array( [obj.word for obj in list_of_words] )
    # check to make sure all the words are the same length
    self.grid  = np.concatenate( [obj.grid for obj in list_of_words] ).astype(bool)

    # some calculations we will use over and over
    self.letters = self.grid.any(axis=1) # letters in word

  def copy(self):
    new_dictionary = Dictionary()
    new_dictionary.words = self.words.copy()
    new_dictionary.grid  = self.grid.copy()

  def guess(self, s):
    w = Word(s)

    # 1. Check if any letters are matching:
    new_grid      = w.grid & self.grid
    found_letters = new_grid.max(axis=2)
    nfl           = np.logical_not(found_letters)
    new_grid[nfl] = True

    # 2. Check what letters are present in both
    in_guess           = w.grid.any(axis=1)
    in_dict            = self.grid.any(axis=1)
    in_both            = in_guess & in_dict
    not_in             = in_guess & np.logical_not( in_dict )
    new_grid           = new_grid.swapaxes(1, 2)
    new_grid[ not_in ] = False
    new_grid           = new_grid.swapaxes(1, 2)

    self.new_grid      = new_grid

    # 3. What can we kill?
    n_to_kill = 0
    for i in range(7,  len(self.words) ):
      info_slice = new_grid[i:(i + 1)]
      
      # 3.1. kill based on whether the potential word has a letter we know is
      #      not in the word
      kill_1 = (np.logical_not( info_slice ) & self.grid).any(axis=2).any(axis=1)

      # 3.2  Assure all letters we know we have, based on the guess, are
      #      in all potential words.
      kill_2 = np.logical_and( in_both, self.letters).any(axis=1)

      kill = np.logical_or(kill_1, kill_2)
      n_to_kill += kill.sum()

    return n_to_kill / len( self.words )

word_list = [Word(obj) for obj in flw_list]
word_dict = Dictionary(word_list)

max_killed = 0
best_word = ''
i = 0
scores = {}

for base_word in word_dict.words:
  score = word_dict.guess(base_word)
  if score > max_killed:
    best_word = base_word
    max_killed = score
  i += 1
  scores[ base_word ] = score
  
  print('base word: {a}; best word so far: {b}; words completed: {c}; words remaining: {d}, killed: {e}'.format(a=base_word, 
                                                                                                   b=best_word, 
                                                                                                   c=i,
                                                                                                   d=(len(flw_list) - i),
                                                                                                   e=score)
  )

## display results
df = pd.DataFrame( [scores.keys(), scores.values()] ).T
df.columns = ['Word', 'Word Score']
df.index = df['Word']
df = df.sort_values( by = 'Word Score' )[['Word Score']]
df.to_csv( 'word_scores.csv' )

print( df )

