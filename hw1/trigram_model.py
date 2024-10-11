import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    n_grams = []
    if n == 1:
        sequence = ['START'] + sequence + ['STOP']
    else:
        sequence = ['START'] * (n-1) + sequence + ['STOP']
    for i in range(len(sequence) - n + 1):
        n_grams.append(tuple(sequence[i:i+n]))
    return n_grams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {}
        self.trigramcounts = {}

        ##Your code here
        self.total_word = 0
        for seq in corpus:
            n_list, dict_list = [1, 2, 3], [self.unigramcounts, self.bigramcounts, self.trigramcounts]
            for i in range(len(n_list)):
                n_grams = get_ngrams(seq, n_list[i])
                gram_dict = dict_list[i]
                for n_gram in n_grams:
                    if n_gram in gram_dict:
                        gram_dict[n_gram] += 1
                    else:
                        gram_dict[n_gram] = 1
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        context = trigram[:-1]
        numerator = self.trigramcounts[trigram] if trigram in self.trigramcounts else 0
        if context == ('START', 'START'):
            denominator = self.unigramcounts[('START',)]
        else:
            denominator = self.bigramcounts[context] if context in self.bigramcounts else 0
        prob = 1 / len(self.lexicon) if denominator == 0 else numerator / denominator
        return prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        context = bigram[:-1]
        numerator = self.bigramcounts[bigram] if bigram in self.bigramcounts else 0
        denominator = self.unigramcounts[context] if context in self.unigramcounts else 0
        prob = 1 / len(self.lexicon) if denominator == 0 else numerator / denominator
        return prob
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        if self.total_word == 0:
            for word in self.lexicon:
                self.total_word += self.unigramcounts[(word, )]
        prob = self.unigramcounts[unigram] / self.total_word
        return prob

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = ["START"] * 2
        for i in range(t):
            context = tuple(result[-2:])
            trigram_probs = defaultdict()
            for word in self.lexicon:
                if word == "START":
                    trigram_probs[word] = 0
                    continue
                trigram = (*context, word)
                trigram_probs[trigram] = self.raw_trigram_probability(trigram)
            selection = random.choices(list(trigram_probs.keys()), weights=list(trigram_probs.values()), k=1)
            result.append(selection[0][-1])
            if result[-1] == "STOP":
                break
        return result[2:]

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        # print(trigram[2:], trigram)
        # print(self.raw_unigram_probability(trigram[2:]), self.raw_bigram_probability(trigram[1:]), self.raw_trigram_probability(trigram))
        prob = lambda1 * self.raw_unigram_probability(trigram[2:]) + \
            lambda2 * self.raw_bigram_probability(trigram[1:]) + \
            lambda3 * self.raw_trigram_probability(trigram)
        return prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        prob_logits = 0
        n = 3
        for trigram in get_ngrams(sentence, n):
            prob = self.smoothed_trigram_probability(trigram)
            prob_logits += math.log2(prob)
        return prob_logits

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        logits = 0
        total_words = 0
        for sentence in corpus:
            logits += self.sentence_logprob(sentence)
            total_words += len(sentence)
        return 2 ** (-logits / total_words)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # ..
            # high corpus
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            correct += 1 if pp < pp2 else 0
            total += 1

        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # ..
            # low corpus
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            correct += 1 if pp < pp2 else 0
            total += 1
        
        return correct / total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity:
    train_corpus = corpus_reader(sys.argv[1], model.lexicon)
    pp = model.perplexity(train_corpus)
    print(f"Training Perplexity: {pp}")

    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(f"Dev Perplexity: {pp}")


    # Essay scoring experiment:
    dir = "./hw1_data/ets_toefl_data/"
    acc = essay_scoring_experiment(dir + 'train_high.txt', dir + 'train_low.txt', dir + "test_high", dir + "test_low")
    print(f"Classification Accuracy: {acc}")

