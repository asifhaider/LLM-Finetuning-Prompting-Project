from smooth_bleu import bleu_fromstr
#  read a gold file
gold_file = open("ref-5000-gold.txt")
# read a pred file
pred_file = open("ref-5000-pred.txt")

# read all the lines in a list
gold_list = gold_file.readlines()
pred_list = pred_file.readlines()

# calculate bleu-4
bleu = bleu_fromstr(pred_list, gold_list, rmstop=False)
print(f"BLEU: {bleu}")

# calculate exact match
em = 0
for pred, gold in zip(pred_list, gold_list):
    if " ".join(pred.split()) == " ".join(gold.split()):
        em += 1
em = em / len(gold_list)
print(f"EM: {em}")

# calculate bertscore
from bert_score import score
P, R, F1 = score(cands=pred_list, refs=gold_list, lang="en")

# print(P)
# print(R)
# print(F1)
# BERTScore average F1 upto 4 decimal places
print(f"BERTScore: {F1.mean():.4f}")