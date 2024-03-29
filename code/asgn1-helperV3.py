#Here are some libraries you're likely to use. You might want/need others as well.
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


n = 3 #n for ngram
lambdas = [.1, 0.2 ,.7] #interpolation parameter from unigram to ngram
smooth = 0.000001 #smooth parameter
ngramConditionProbs = collections.defaultdict(dict) #collection of ngram model
ngramCounts = defaultdict(float) #collection of counts
conditionProbs = collections.defaultdict(dict) #final ngram model
ntypes = len('0qwertyuiopasdfghjklzxcvbnm ].,')
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
            if n is not 1:
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
        ngramCounts[str] +=  smooth*ntypes
        prob = defaultdict(float)
        for k in '0qwertyuiopasdfghjklzxcvbnm ].,':
            prob[k] =  smooth/ngramCounts[str]
        ngramConditionProbs[str]  = prob
    else:
        for i in '[0qwertyuiopasdfghjklzxcvbnm .,': 
               initialConditions(str+i, n-1)
                                   
#do interpolation of probability, given manuel selected parameter
def readjustProbability(str, n,N):
    if n == 1:
        for k in '0qwertyuiopasdfghjklzxcvbnm ].,':
            conditionProbs[str][k] = lambdas[0]*ngramConditionProbs[''][k]
            for i in range(N-1):
                conditionProbs[str][k] += lambdas[i+1]*ngramConditionProbs[str[-i+1:]][k]
      #      print conditionProbs[str][k]
    else:
        for i in '[0qwertyuiopasdfghjklzxcvbnm .,': 
               readjustProbability(str+i, n-1,N)

                                   
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
                for i in range(n):
                    ngram = line[j+n-i-1:j+n]
                    ngramCounts[ngram] +=1
    for c in '0qwertyuiopasdfghjklzxcvbnm ].,':
        ngramCounts[''] += ngramCounts[c]
    for i in range(n):
        initialConditions('', i+1)

   # print conditionProbs
            
    for ngram in ngramCounts.keys():
        if ngram is not '': 
            ngramConditionProbs[ngram[0:-1]][ngram[-1]] +=  ngramCounts[ngram]/ngramCounts[ngram[0:-1]]
            
    
    readjustProbability('',n,n)
        
    condition = 'an'
    print "Conditional probability for "+condition
    for p in sorted(conditionProbs[condition].keys()):
        print 'P('+p+'|'+condition+')='+str(conditionProbs[condition][p])
        
    json.dump(conditionProbs, open(infile+'.out3','w'))

    print "\nRandom Text"
    random = generate_random_output(conditionProbs, 300)
    with open(infile+'.random3', "w") as text_file:
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
