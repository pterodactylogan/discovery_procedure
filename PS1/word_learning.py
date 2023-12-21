import random
import pandas as pd
import numpy as np
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

# set up training data
data = open("rollins.txt", "r")
instances = data.read().split("\n\n")

    
# set up gold lexicon
gold = open("gold.txt", "r")
gold_lex = {}
lines = gold.readlines()
for line in lines:
    pair = line.split()
    gold_lex[pair[0]] = pair[1]

# helper functions
def increment(count_dict, item):
    if item not in count_dict:
        count_dict[item] = 1
    else:
        count_dict[item] += 1

def prob(meaning, word, assocs):
    num = assocs[meaning][word] + SMOOTH
    den = assocs.loc[word].sum() + (assocs.shape[1] * SMOOTH)
    return num/den

# ALGORITHMS
def propose_but_verify():
    lexicon = {}

    for point in instances:
        pair = point.split("\n")
        words = pair[0].split()
        referents = pair[1].split()

        for word in words:
            if (word not in lexicon) or lexicon[word] not in referents:
                lexicon[word] = random.choice(referents)

    data.close()
    return lexicon

def global_xsit():
    counts = {}
    lexicon = {}
    for point in instances:
        pair = point.split("\n")
        words = pair[0].split()
        referents = pair[1].split()

        for word in words:
            max_count = 0
            max_refs = []
            if word not in counts:
                counts[word] = {}
            for item in referents:
                increment(counts[word], item)
                if(counts[word][item] > max_count):
                    max_count = counts[word][item]
                    max_refs = [item]
                if(counts[word][item] == max_count):
                    max_refs.append(item)
            lexicon[word] = random.choice(max_refs)
        
    return lexicon


def pursuit(lrate, smooth, theta, sampling = False):
    assocs = pd.DataFrame()
    lexicon = {}
    for point in instances:
        pair = point.split("\n")
        words = pair[0].split()
        referents = pair[1].split()

        for word in words:
            # INITIALIZE
            if word not in assocs.index:
                min_assoc = 1
                min_refs = []
                for ref in referents:
                    if ref not in assocs.columns:
                        if min_assoc == 0:
                            min_refs.append(ref)
                        else:
                            min_assoc = 0
                            min_refs = [ref]
                    else:
                        max_assoc = assocs[ref].max()
                        if max_assoc < min_assoc:
                            min_assoc = max_assoc
                            min_refs = [ref]
                        if max_assoc == min_assoc:
                            min_refs.append(ref)

                h0 = random.choice(min_refs)
                if h0 not in assocs.columns:
                    # initialize meaning with 0 across the board
                    assocs[h0] = [0 for i in range(assocs.shape[0])]
                
                # initialize word with 0 assocs acros the board
                assocs.loc[word] = [0 for i in range(assocs.shape[1])]
                
                assocs[h0][word] = LRATE

            # ADJUST
            else:    
                # select most favored hypothesis
                if not sampling:
                    max_score = 0
                    max_hs = []
                    for h in assocs.columns:
                        if assocs[h][word] == max_score:
                            max_hs.append(h)       
                        if assocs[h][word] > max_score:
                            max_score = assocs[h][word]
                            max_hs = [h]

                    selected = random.choice(max_hs)
                else:
                    # sample according to weights
                    x = np.arange(0, len(assocs.columns))
                    weights = assocs.loc[word]
                    weights /= weights.sum()
                    distro = stats.rv_discrete(values = (x,
                                                         weights))
                    selected = assocs.columns[distro.rvs()]
                
                if selected in referents:
                    # increase score
                    assocs[selected][word] += (LRATE*(1 - assocs[selected][word]))
                    if prob(selected, word, assocs) > THETA:
                        lexicon[word]=selected
                else:
                    # decrease score
                    assocs[selected][word] *= (1-LRATE)
                    # pick new hypothesis at random from avaialable refs
                    new_h = random.choice(referents)
                    if new_h not in assocs.columns:
                        assocs[new_h] = [0 for i in range(assocs.shape[0])]
                        assocs[new_h][word] = LRATE
                    else:
                        a = assocs[new_h][word]
                        assocs[new_h][word] = a + (LRATE*(1-a))
                    if prob(new_h, word, assocs) > THETA:
                        lexicon[word]=new_h

    return lexicon



LRATE = 0.02
SMOOTH = 0.05
THETA = 0.15

##ps = []
##rs = []
##fs = []
##for i in range(20):
##    result = pursuit(LRATE, SMOOTH, THETA, True)
##    num_correct = 0
##    for word in gold_lex:
##        if word in result:
##            if result[word] == gold_lex[word]:
##                num_correct += 1
##
##    precision = (num_correct / len(result))
##    recall = (num_correct / len(gold_lex))
##    f1 = 2 / ((1 / precision) + (1/recall))
##    ps.append(precision)
##    rs.append(recall)
##    fs.append(f1)
##
##print("p:", sum(ps)/len(ps))
##print("r:", sum(rs)/len(rs))
##print("f1:", sum(fs)/len(fs))

random.shuffle(instances)

result = propose_but_verify()

num_correct = 0
for word in gold_lex:
    if word in result:
        if result[word] == gold_lex[word]:
            num_correct += 1


precision = (num_correct / len(result))
recall = (num_correct / len(gold_lex))
f1 = (2 / ((1 / precision) + (1/recall)))

print("Propose but Verify:")
print("p:", precision)
print("r:", recall)
print("f1:", f1)

result = global_xsit()

num_correct = 0
for word in gold_lex:
    if word in result:
        if result[word] == gold_lex[word]:
            num_correct += 1


precision = (num_correct / len(result))
recall = (num_correct / len(gold_lex))
f1 = (2 / ((1 / precision) + (1/recall)))

print("Global Cross-Situational:")
print("p:", precision)
print("r:", recall)
print("f1:", f1)

result = pursuit(LRATE, SMOOTH, THETA)

num_correct = 0
for word in gold_lex:
    if word in result:
        if result[word] == gold_lex[word]:
            num_correct += 1


precision = (num_correct / len(result))
recall = (num_correct / len(gold_lex))
f1 = (2 / ((1 / precision) + (1/recall)))

print("Pursuit:")
print("p:", precision)
print("r:", recall)
print("f1:", f1)

result = pursuit(LRATE, SMOOTH, THETA, True)

num_correct = 0
for word in gold_lex:
    if word in result:
        if result[word] == gold_lex[word]:
            num_correct += 1


precision = (num_correct / len(result))
recall = (num_correct / len(gold_lex))
f1 = (2 / ((1 / precision) + (1/recall)))

print("Pursuit with Sampling:")
print("p:", precision)
print("r:", recall)
print("f1:", f1)


