from __future__ import division
import re
import collections
import sys
import json
from random import random
from math import log, log10
from collections import defaultdict
import numpy as np #numpy provides useful maths and vector operations
from numpy.random import random_sample


tri_counts=defaultdict(int) #counts of all trigrams in input
uni_counts=defaultdict(int) #counts of all trigrams in input
bi_counts=defaultdict(int) #counts of all trigrams in input
n = 1 #ngram model
smooth = .1 #smooth parameter
nrange = '0qwertyuiopasdfghjklzxcvbnm ].,'
ntypes = len(nrange)
pairsCounts = defaultdict(float)
conditionProbs = collections.defaultdict(dict)
#function turns input into required format
def preprocess_line(line):
    #remove non-necessary characters, 
    #and turn string to lowercase
    p = re.compile('[^\w\s,.]')
    line = re.sub(p,'',line.lower())
    #replace \n by ]
    line = re.sub('\n',']',line)
    #turn numbers into 0
    line = re.sub('[0-9]','0',line)
    #add begining and end [[
    for i in range(1, n):
        line = '['+line
    return line

# this function generate random output from the estimated language models.
def generate_random_output(distribution, N):
    ''' generate_random_sequence takes a distribution (represented as a
    dictionary of outcome-probability pairs) and a number of samples N
    and returns a list of N samples from the distribution.  
    This is a modified version of a sequence generator by fraxel on
    StackOverflow:
    http://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
    '''
    #As noted elsewhere, the ordering of keys and values accessed from
    #a dictionary is arbitrary. However we are guaranteed that keys()
    #and values() will use the *same* ordering, as long as we have not
    #modified the dictionary in between calling them.
    output = '';
    for i in range(1, N):
        if len(output) < n-1 or (output[-1:] is ']'):
            for i in range(1, n):
                output += '['
            continue 
        
        if n == 1:
            nextIndex = ''
        else:
            nextIndex = output[-(n-1):]

        outcomes = np.array(distribution[nextIndex].keys())
        probs = np.array(distribution[nextIndex].values())
        # print probs
        #make an array with the cumulative sum of probabilities at each
        #index (ie prob. mass func)
        bins = np.cumsum(probs)
        #create N random #s from 0-1
        #digitize tells us which bin they fall into.
        #return the sequence of outcomes associated with that sequence of bins
        #(we convert it from array back to list first)

        output += outcomes[np.digitize(random_sample(1), bins)][0]

    p = re.compile('[\[]')
    return re.sub('\]','\n',re.sub(p,'',output))

def initialConditions(str, n):
    if n == 1:
        pairsCounts[str] +=  smooth*ntypes
        prob = defaultdict(float)
        for k in nrange:
            prob[k] =  smooth/pairsCounts[str]
            conditionProbs[str]  = prob
    else:
        for i in nrange: 
               initialConditions(str+i, n-1)
    
def calculate_perplexity(tokens, probs, n):
    entropy = 0.0
    for token in tokens:
        # please comment this line after implementing smooth method
        # if probs.get(token[0:len(token)-1]) is not None and probs.get(token[0:len(token)-1]).get(token[len(token)-1]) is not None:
          #  print token
            entropy -= log10(probs.get(token[0:len(token)-1]).get(token[len(token)-1]))

    return 10**(entropy / len(tokens))

#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.

if len(sys.argv) < 3:
    print "Usage: ", sys.argv[0], "<mode> <input_file> <testing_file>"
    sys.exit(0)

mode = sys.argv[1] #get input argument: training or testing mode
infile = sys.argv[2] #get input argument: the training file

if mode == 'train':
    if len(sys.argv) != 3:
        print "Usage: ", sys.argv[0], "<mode> <training_file> <testing_file>"
        sys.exit(0)
    with open(infile) as f:
        for line in f:
            line = preprocess_line(line) #doesn't do anything yet.
            #our model unit is a line instead of sentence
            for j in range(len(line)-(n-1)):
                trigram = line[j:j+n]
                tri_counts[trigram] += 1

    # smooth method
    for trigram in tri_counts.keys():
        pairsCounts[trigram[0:n-1]] +=  tri_counts[trigram]

    initialConditions('', n)
    
    for trigram in tri_counts.keys():
        conditionProbs[trigram[0:n-1]][trigram[n-1]] +=  tri_counts[trigram]/pairsCounts[trigram[0:n-1]]

    print conditionProbs

    # condition = 'an'
    # print "Conditional probability for "+condition
    # for p in sorted(conditionProbs[condition].keys()):
    #     print 'P('+p+'|'+condition+')='+str(conditionProbs[condition][p])
        
    json.dump(conditionProbs, open(infile+'.out','w'))

    print "\nRandom Text"
    random = generate_random_output(conditionProbs, 300)
    with open(infile+'.random', "w") as text_file:
        text_file.write(random)
    print random
elif mode == 'test':
    if len(sys.argv) != 4:
        print "Usage: ", sys.argv[0], "<mode> <model_file> <testing_file>"
        sys.exit(0)
    
    testfile = sys.argv[3]
    wordlist = []

    with open(infile) as f:
        conditionProbs = json.load(f)
    # redifine n-gram model
    n = len(conditionProbs.keys()[0]) + 1
    
    with open(testfile) as f:
        for line in f:
            line = preprocess_line(line) #doesn't do anything yet.
            #our model unit is a line instead of sentence
            for j in range(len(line)-(n-1)):
                wordlist.append(line[j:j+n])
    print "Perplexity of the testing data based on the input model:"      
    print calculate_perplexity(wordlist, conditionProbs, n);
else:
    print "Running mode should be either <train> or <test>";

#totalCounts = 0
#for counts in uni_counts.values():
#    totalCounts += counts
#prob = defaultdict(float)
#for uni in uni_counts.keys():
#    prob[uni] = uni_counts[uni]/totalCounts
#conditionProbs['[['] = prob
#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
#print "Trigram counts in ", infile, ", sorted alphabetically:"
#for trigram in sorted(tri_counts.keys()):
#    print trigram, ": ", tri_counts[trigram]
#print "Trigram counts in ", infile, ", sorted numerically:"
#for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
#    print tri_count[0], ": ", str(tri_count[1])
