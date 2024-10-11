"""
COMS W4705 - Natural Language Processing - Fall 2024
PCFG parser ungraded exercise
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        table = {}
        n = len(tokens)
        # initializion
        for i in range(n): 
          table[(i,i+1)] = set()
          try: 
            token = (tokens[i],)
            rules = self.grammar.rhs_to_rules[token]
            for lhs, rhs, probs in rules: 
              table[(i,i+1)].add(lhs)
          except KeyError: 
            pass

        # main loop
        for length in range(2,n+1): 
          for i in range(0, n-length+1): 
            j = i+length

            table[(i,j)] = set()

            for k in range(i+1, j):  
              left_nts = table[(i,k)]
              right_nts = table[(k,j)] 
              for B in left_nts:              
                for C in right_nts: 
                  if (B,C) in self.grammar.rhs_to_rules: 
                    rules = self.grammar.rhs_to_rules[(B,C)]
                    for lhs,rhs,prob in rules: 
                      table[(i,j)].add(lhs)                                    

        return grammar.startsymbol in table[(0,n)]
       

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        table, probs = {}, {}

        # TODO
        
        return table, probs 
       


def get_tree(table, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    if (j-i) == 1:           # base case, just return the token.
      return (nt, table[(i,j)][nt])
    
    bp_left, bp_right = table[(i,j)][nt]
    B,i,k = bp_left
    C,k,j = bp_right
    left_subtree = get_tree(table, i, k, B)  # recursively obtain left subtree
    right_subtree = get_tree(table, k, j, C) # and right subtree
    return (nt, left_subtree, right_subtree)

       
if __name__ == "__main__":
    
    with open('testgrammar.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks = "she saw the cat with glasses".split()

        print(parser.is_in_language(toks))
        #table,probs = parser.parse_with_backpointers(toks)
        #tree = get_tree(table, 0, len(toks), grammar.startsymbol)
        #print(tree)
        
