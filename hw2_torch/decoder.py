import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State, RootDummy
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        # print(f"words: {words}")
        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 5
        while state.buffer:
            inputs = self.extractor.get_input_representation(words, pos, state)
            inputs = torch.LongTensor(inputs).unsqueeze(0)
            predictions = self.model(inputs)
            predictions = predictions.detach().numpy().flatten()
            sorted_indices = np.argsort(predictions)[::-1]
            # print(f"stack: {state.stack}, buffer: {state.buffer}")

            for index in sorted_indices:
                transition, label = self.output_labels[index]
                # print(f"transition: {transition}, label: {label}")
                if transition == "shift":
                    # shift if the buffer has more than one word, when the buffer has only one word, we can shift only if the stack is empty
                    if (len(state.buffer) == 1 and len(state.stack) == 0) or len(state.buffer) > 1:
                        state.shift()
                        break
                elif transition == "left_arc":
                    # If the stack is empty, we can't left-arc
                    if state.stack and state.stack[-1] != 0:
                        state.left_arc(label)
                        break
                elif transition == "right_arc":
                    # If the stack is empty, we can't right-arc
                    if state.stack:
                        state.right_arc(label)
                        break

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
