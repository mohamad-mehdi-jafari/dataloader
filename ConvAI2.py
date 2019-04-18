"""
    In this script we're going to perform download, unzip, and create word vectors
    Data set: South Park Dialogue

"""

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext import vocab


tokenize = lambda x: x.split()


LINE = Field(sequential=True, tokenize=tokenize, lower=True)

SEASON = Field(sequential=False, use_vocab=False)
EPISODE = Field(sequential=False, use_vocab=False)
CHARACTER = Field(sequential=False, use_vocab=False)

field_list = [("Season", SEASON), ("Episode", EPISODE),
                 ("Character", CHARACTER), ("Line", LINE)]

data = TabularDataset(
               path="/home/mehdi/Downloads/All-seasons.csv", # path of the file
               format='csv',
               skip_header=True, # if your csv file has header line!
               fields=field_list)

vec = vocab.Vectors('/home/mehdi/Downloads/glove.6B.50d.txt', './data/glove_embedding/')
print(len(data))
LINE.build_vocab(data, vectors=vec)
print(LINE.vocab.vectors.shape)
print(LINE.vocab.itos[10])