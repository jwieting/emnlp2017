import io
import argparse
import random
import sys

def get_length(t):
    l1 = len(t.split())
    if l1 >= 100:
        l1 = 100
    return l1

def get_overlap(t, r, type):
    t = t.split()
    r = r.split()
    if type == 2:
        temp = []
        for i in range(len(t) - 1):
            temp.append(t[i] + " " + t[i + 1])
        t = temp
        temp = []
        for i in range(len(r) - 1):
            temp.append(r[i] + " " + r[i + 1])
        r = temp
    elif type == 3:
        temp = []
        for i in range(len(t) - 2):
            temp.append(t[i] + " " + t[i + 1] + " " + t[i + 2])
        t = temp
        temp = []
        for i in range(len(r) - 2):
            temp.append(r[i] + " " + r[i + 1] + " " + r[i + 2])
        r = temp
    if len(r) < len(t):
        start = r
        end = t
    else:
        start = t
        end = r
    start = list(start)
    den = len(start)
    if den == 0:
        return 0.
    num = 0
    for i in range(len(start)):
        if start[i] in end:
            num += 1
    return float(num) / den

parser = argparse.ArgumentParser()
parser.add_argument("-outfile", help="Name of output file")
parser.add_argument("-infile", help="Name of input file")
parser.add_argument("-filtering_method", help="Approach to filter the data, either trans, length, ovl-1, ovl-2, ovl-3")
args = parser.parse_args()

lines = io.open(args.infile, 'r', encoding='utf-8').readlines()
examples = []
currg = None
currt = []
idx = 0
for n,i in enumerate(lines):
    if n == 0:
        currg = i.strip()
    elif len(i.strip()) == 0:
        random.shuffle(currt)
        examples.append((currg, currt, idx))
        currg = None
        currt = []
        idx += 1
    elif len(i.strip()) > 0 and not currg:
        currg = i.strip()
    else:
        currt.append(i.strip())

fout = io.open(args.outfile, 'w', encoding='utf-8')
for i in examples:
    gold = i[0]
    idx = i[2]
    for j in i[1]:
        j = j.split('\t')
        sent = j[0].strip()
        ts = j[1]
        score = None
        if args.filtering_method == "trans":
            score = float(ts)
        elif args.filtering_method == "length":
            score = get_length(sent)
        elif args.filtering_method == "ovl-1":
            score = get_overlap(gold, sent, 1)
        elif args.filtering_method == "ovl-2":
            score = get_overlap(gold, sent, 2)
        elif args.filtering_method == "ovl-3":
            score = get_overlap(gold, sent, 3)
        else:
            "Please enter a valid filtering method. Exiting."
            sys.exit(0)
        ln = "{0}\t{1}\t{2}\t{3}\n".format(gold, sent, score, idx)
        fout.write(unicode(ln))
fout.close()
