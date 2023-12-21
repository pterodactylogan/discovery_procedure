from wordseg.algos.tp import segment
from wordseg.evaluate import evaluate, summary

stimulus_file = open("stimulus_tp0.txt", "r")
gold_file = open("gold0.txt", "r")

result = segment(stimulus_file.readlines())

gold = gold_file.read().splitlines()

print(evaluate(result, gold))

stimulus_file.close()
gold_file.close()
