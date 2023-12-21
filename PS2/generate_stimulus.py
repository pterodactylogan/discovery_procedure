import random
import scipy.stats as stats
import numpy as np
import re

lex_file = open("./lexicon0.txt", "w")
stim_file_tp = open("./stimulus_tp0.txt", "w")
stim_file_sub = open("./stimulus_sub0.txt", "w")
gold_file = open("./gold0.txt", "w")

# 24 consonants
consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
              'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'z',
              'θ', 'ð', 'ʃ', 'ʒ']

# 14 vowels
vowels = ['a','e','i','o','u', 'ɛ', 'ɐ', 'ɪ',
          'ə', 'y', 'ʊ', 'ʌ', 'ɔ', 'ø']

# create 1,000-word lexicon
lexicon = []
for i in range(1000):
    num_syllables = stats.poisson.rvs(2) + 1
    word = ""
    for j in range(num_syllables):
        word += random.choice(consonants) + random.choice(vowels) + " "
    lexicon.append(word)

# store it in lexicon file
lex_file.write("\n".join(lexicon))

# create zipfian distro over lexicon
n = len(lexicon)
x = np.arange(1, n+1)
weights = 1 / x
weights /= weights.sum()
bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))


# create 15,000 sentences
sentences = []
segmented_sentences = []
sub_sentences = []
for i in range(15000):
    num_words = stats.poisson.rvs(2) + 1
    sentence = ""
    segmented_sentence = ""
    sub_sentence = ""
    prev_word_rank = -1
    for j in range (num_words):
        # select word according to zipfian distro
        while True:
            rank = bounded_zipf.rvs()
            if rank != prev_word_rank:
                word = lexicon[rank-1]
                sentence += word
                segmented_sentence += re.sub(" ", "", word) + " "
                x = ".".join(word.strip())
                sub_sentence += re.sub(". .", "|", x) + " "
                prev_word_rank = rank
                break
        
    sentences.append(sentence)
    segmented_sentences.append(segmented_sentence.strip())
    sub_sentences.append(sub_sentence)

stim_file_tp.write("\n".join(sentences))
gold_file.write("\n".join(segmented_sentences))
stim_file_sub.write("\n".join(sub_sentences))

lex_file.close()
stim_file_tp.close()
gold_file.close()
stim_file_sub.close()
