import string
import numpy as np

from vocabulary import Vocabulary
from collections import Counter

class ReviewVectorizer(object):
    """ The vectorizer class which coordinates the Vocabularies and puts them to use """
    
    def __init__(self, review_vocab, sentiment_vocab):
        """
        Args:
            review_vocab (Vocabulary): maps words to integers
            sentiment_vocab (Vocabulary): maps class labels to integers
        """

        self.review_vocab = review_vocab
        self.sentiment_vocab = sentiment_vocab

    def vectorize(self, review):
        """ Create a collapsed one­hit vector for the review

        Args:
            review (str): the review

        Returns:
            one_hot (np.ndarray): the collapsed one­hot encoding
        """

        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """ Instantiate the vectorizer from the dataset dataframe

        Args:
            review_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency­based filtering

        Returns: 
            an instance of the ReviewVectorizer
        """

        review_vocab = Vocabulary(add_unk=True)
        sentiment_vocab = Vocabulary(add_unk=False)

        # Add sentiments
        for sentiment in sorted(set(review_df.sentiment)):
            sentiment_vocab.add_token(sentiment)

        # Add top words if count > provided count
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, sentiment_vocab)