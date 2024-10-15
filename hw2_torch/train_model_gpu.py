import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor


class DependencyDataset(Dataset):

  def __init__(self, input_filename, output_filename):
    self.inputs = np.load(input_filename)
    self.outputs = np.load(output_filename)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

  def __init__(self, word_types, outputs):
    super(DependencyModel, self).__init__()
    # TODO: complete for part 3
    self.embed = Embedding(word_types, 128)
    self.fc1 = Linear(768, 128)
    self.fc2 = Linear(128, outputs)

  def forward(self, inputs):
    # TODO: complete for part 3
    embeds = self.embed(inputs)
    embeds = embeds.view(-1, 768)
    logits = self.fc2(relu(self.fc1(embeds)))
    return logits  # replace this line


def train(model, loader, device):
  # loss_function = NLLLoss(reduction='mean')
  loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

  model.to(device)

  LEARNING_RATE = 0.01 
  optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

  tr_loss = 0 
  tr_steps = 0

  # put model in training mode
  model.train()
 

  correct = 0 
  total =  0 
  for idx, batch in enumerate(loader):
 
    inputs, targets = batch
    inputs, targets = inputs.to(device).to(torch.int64), targets.to(device)

    predictions = model(torch.cuda.LongTensor(inputs))
    loss = loss_function(predictions, targets)
    tr_loss += loss.item()

    # print("Batch loss: ", loss.item()) # Helpful for debugging, maybe

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")

    # To compute training accuracy for this epoch 
    correct += sum(torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1))
    total += len(inputs)

    # Run the backward pass to update parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  epoch_loss = tr_loss / tr_steps
  acc = correct / total
  print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader, device)


    torch.save(model.state_dict(), sys.argv[3]) 
