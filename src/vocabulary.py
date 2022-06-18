class Vocabulary(object):
    """ Class to process text and extract Vocabulary for mapping """

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>", add_pad=True, pad_token="<PAD>"):
        """
        Args:
            token_to_idx (dict): a pre-­existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token
        self.pad_token = pad_token

        if add_pad:
            self.add_token(pad_token)
        
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    @property
    def token_to_idx(self):
        return self.__token_to_idx

    @token_to_idx.setter
    def token_to_idx(self, value):
        self.__token_to_idx = value

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        if not isinstance(value, str):
            raise ValueError("[!!] Required string parameter --> Passed was: {}".format(type(value)))
        self._pad_token = value

    def add_token(self, token):
        """ Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary

        Returns:
            index (int): the integer corresponding to the token
        """

        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index

    def lookup_token(self, token):
        """ Retrieve the index associated with the token
            or the UNK index if token isn't present.

        Args:
            token (str): the token to look up

        Returns:
            index (int): the index corresponding to the token
        
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
            for the UNK functionality
        """

        if self._add_unk:
            return self.token_to_idx.get(token, self.unk_index)
        else:
            return self.token_to_idx[token]
    
    def lookup_index(self, index):
        """ Return the token associated with the index
        
        Args:
            index (int): the index to look up
        
        Returns:
            token (str): the token corresponding to the index

        Raises:
            KeyError: if the index is not in the Vocabulary
        """

        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)
    
    def __len__(self):
        return len(self.token_to_idx)