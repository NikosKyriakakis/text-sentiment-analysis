from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from vectorizer import ReviewVectorizer

import pandas as pd

class ReviewDataset(Dataset):
    def __init__(self, review_data, vectorizer):
        """
        Args:
            review_data (pandas.DataFrame): the dataset
            vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """

        self.review_data = review_data
        self._vectorizer = vectorizer
        self.train_data, self.test_data = train_test_split(self.review_data, test_size=0.2)
        self.train_data, self.val_data = train_test_split(self.train_data, test_size=0.25)

        self._lookup_dict = {
            'train': (self.train_data, len(self.train_data)),
            'val': (self.val_data, len(self.val_data)),
            'test': (self.test_data, len(self.test_data))
        }

        self.set_split('train')

    def generate_batches(self, batch_size, shuffle=True, drop_last=True, device="cpu"):
        dataloader = DataLoader (
            dataset=self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            drop_last=drop_last
        )

        for data_dict in dataloader:
            out_data_dict = {}
            for name, _ in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict


    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """ Load dataset and make a new vectorizer from scratch

        Args:
            review_csv (str): location of the dataset

        Returns:
            an instance of ReviewDataset
        """

        review_data = pd.read_csv(review_csv)
        return cls(review_data, ReviewVectorizer.from_dataframe(review_data))

    def get_vectorizer(self):
        """ Returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ Selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of "train", "val", or "test"
        """

        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """ The primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
            
        Returns:
            a dict of the data point's features (x_data) and label (y_target)
        """
        
        row = self._target_df.iloc[index]

        review_vector = \
            self._vectorizer.vectorize(row.review)

        sentiment_index = \
            self._vectorizer.sentiment_vocab.lookup_token(row.sentiment)

        return {'x_data': review_vector, 'y_target': sentiment_index}