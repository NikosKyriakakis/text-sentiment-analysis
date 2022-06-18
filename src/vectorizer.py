import string
import numpy as np

from vocabulary import Vocabulary
from collections import Counter
from abc import ABC, abstractmethod

class TextVectorizer(ABC):
    """ The vectorizer class which coordinates the Vocabularies and puts them to use """
    
    def __init__(self, text_vocab, label_vocab, seq_len=0):
        """
        Args:
            text_vocab (Vocabulary): maps words to integers
            label_vocab (Vocabulary): maps class labels to integers
        """

        self.text_vocab = text_vocab
        self.label_vocab = label_vocab

    @abstractmethod
    def vectorize(self, text):
        """ Empty abstract method """


    @classmethod
    def from_dataframe(cls, text_data, cutoff=25, seq_len=500):
        """ Instantiate the vectorizer from the dataset dataframe

        Args:
            text_data (pandas.DataFrame): the text dataset
            cutoff (int): the parameter for frequency­based filtering

        Returns: 
            an instance of the TextVectorizer
        """

        text_vocab = Vocabulary(add_unk=True)
        label_vocab = Vocabulary(add_unk=False)

        # Add labels
        for label in sorted(set(text_data.label)):
            label_vocab.add_token(label)

        # Add top words if count > provided count
        word_counts = Counter()
        for text in text_data.text:
            for word in text.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                text_vocab.add_token(word)

        return cls(text_vocab, label_vocab, seq_len)

class OneHotVectorizer(TextVectorizer):
    def __init__(self, text_vocab, label_vocab, seq_len):
        super().__init__(text_vocab, label_vocab, seq_len)
    
    def vectorize(self, text):
        """ Create a collapsed one­hit vector for the text
        Args:
            text (str): the text
        Returns:
            one_hot (np.ndarray): the collapsed one­hot encoding
        """

        one_hot = np.zeros(len(self.text_vocab), dtype=np.float32)
        for token in text.split(" "):
            if token not in string.punctuation:
                one_hot[self.text_vocab.lookup_token(token)] = 1

        return one_hot

class PaddingVectorizer(TextVectorizer):
    def __init__(self, text_vocab, label_vocab, seq_len):
        super().__init__(text_vocab, label_vocab, seq_len)

        self.seq_len = seq_len

    @property
    def seq_len(self):
        return self.__seq_len

    @seq_len.setter
    def seq_len(self, value):
        if value <= 0:
            raise ValueError("[!!] Invalid sequence length provided. --> Expected non-negative input.")
        self.__seq_len = value

    def vectorize(self, text):
        padded_text = []
        for token in text.split(" "):
            if token not in string.punctuation:
                padded_text.append(self.text_vocab.lookup_token(token))
        padded_text = (self.seq_len - len(padded_text)) * [0] + padded_text

        return padded_text

