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
    sequence = ['START'] * n + sequence + ['STOP']
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
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        for seq in corpus:
            n_list, dict_list = [1, 2, 3], [self.unigramcounts, self.bigramcounts, self.trigramcounts]
            for i in range(len(n_list)):
                n_grams = get_ngrams(seq, n_list[i])
                gram_dict = dict_list[i]
                for n_gram in n_grams:
                    gram_dict[n_gram] += 1
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        context = trigram[:-1]
        numerator = self.trigramcounts[trigram]
        denominator = self.bigramcounts[context]
        prob = 1 / len(self.lexicon) if denominator == 0 else numerator / denominator
        return prob

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        context = bigram[:-1]
        numerator = self.bigramcounts[bigram]
        denominator = self.unigramcounts[context]
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
        self.total_word = 0
        if not self.total_word:
            for word in self.lexicon:
                self.total_word += self.lexicon[word]
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
            context = result[-2:]
            trigram_probs = defaultdict()
            for word in self.lexicon:
                if word == "START":
                    trigram_probs[word] = 0
                    continue
                trigram = (*context, word)
                trigram_probs[trigram] = self.raw_trigram_probability(trigram)
            selection = random.choices(list(trigram_probs.keys()), weights=trigram_probs.values(), k=1)
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
        prob = lambda1 * self.unigramcounts[tuple(trigram[-1])] + \
            lambda2 * self.bigramcounts[tuple(trigram[1:])] + \
            lambda3 * self.trigramcounts[tuple(trigram)]
        return prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        prob_logits = 0
        n = 3
        sentence = ['START'] * n + sentence + ['STOP']
        for i in range(n, len(sentence)):
            trigram = tuple(sentence[i-n:i])
            prob = self.smoothed_trigram_probability(trigram)
            if prob:
                prob_logits += math.log2(prob)
        return prob_logits

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        logits = 0
        for sentence in corpus:
            logits += self.sentence_logprob(sentence)
        logits /= len(self.lexicon)
        return 2 ** (-logits)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # ..
            # high corpus
            test_corpus = corpus_reader(os.path.join(testdir1, f))
            for sentence in test_corpus:
                high_prob = model1.sentence_logprob(sentence)
                low_prob = model2.sentence_logprob(sentence)
                total += 1
                correct += 1 if high_prob > low_prob else 0

        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # ..
            # low corpus
            test_corpus = corpus_reader(os.path.join(testdir2, f))
            for sentence in test_corpus:
                high_prob = model1.sentence_logprob(sentence)
                low_prob = model2.sentence_logprob(sentence)
                total += 1
                correct += 1 if high_prob < low_prob else 0
        
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
    print(pp)

    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment:
    dir = "./hw1_data/ets_toefl_data/"
    acc = essay_scoring_experiment(dir + 'train_high.txt', dir + 'train_low.txt', dir + "test_high", dir + "test_low")
    print(acc)

