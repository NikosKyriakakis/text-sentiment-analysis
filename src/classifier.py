import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.metrics import f1_score

plt.style.use('ggplot')

class MLP(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()

        self._args = args
        self._logs = {
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": []
        }

    def setup(self):
        is_binary = self.__configure_loss_function()
        self.__configure_optimizer()
        self.__configure_metrics_method(is_binary)

    def __configure_loss_function(self):
        if self._args.criterion == "bce_logits":
            is_binary = True
            self._criterion = nn.BCEWithLogitsLoss()
        else:
            is_binary = False
            self._criterion = nn.CrossEntropyLoss()

        return is_binary

    def __configure_optimizer(self):
        if self._args.optimizer == "Adam":
            self._optimizer = optim.Adam(self.parameters(), lr=self._args.learning_rate)
        else:
            self._optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def __configure_metrics_method(self, binary):
        if binary:
            self._compute_accuracy = self._compute_bin_accuracy
            self._compute_f1 = self._compute_bin_f1
            self._prepare_lossfunc = self._prepare_bin_lossfunc
        else:
            self._compute_accuracy = self._compute_mult_accuracy
            self._compute_f1 = self._compute_mult_f1
            self._prepare_lossfunc = self._prepare_mult_lossfunc

    @abstractmethod
    def forward(self, x):
        """ Abstract method """

    def _compute_bin_accuracy(self, y_pred, y_target):
        y_target = y_target.cpu()
        y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    def _compute_bin_f1(self, y_pred, y_target):
        y_target = y_target.cpu()
        y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
        return f1_score(y_target, y_pred_indices, average='macro') * 100

    def _compute_mult_accuracy(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    def _compute_mult_f1(self, y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        return f1_score(y_target, y_pred_indices, average='macro') * 100

    def _prepare_bin_lossfunc(self, tensor_data):
        return tensor_data.view(-1, 1).float()

    def _prepare_mult_lossfunc(self, tensor_data):
        return tensor_data.long()

    def fit(self):
        # Send model to available hardware
        self = self.to(self._args.device)

        for _ in tqdm(range(self._args.num_epochs)):
            loss, acc, f1 = self.train_net()
            self._logs["train_loss"].append(loss)
            self._logs["train_acc"].append(acc)
            self._logs["train_f1"].append(f1)

            loss, acc, f1 = 0, 0, 0
            with torch.no_grad():
                loss, acc, f1 = self.eval_net(mode='val')
                self._logs["val_loss"].append(loss)
                self._logs["val_acc"].append(acc)
                self._logs["val_f1"].append(f1)
        
    def eval_net(self, mode):
        self.eval()

        self._args.dataset.set_split(mode)
        batch_generator = self._args.dataset.generate_batches (
            batch_size=self._args.batch_size, 
            device=self._args.device
        )

        running_loss = 0
        running_acc = 0
        running_f1 = 0

        for batch_index, batch_dict in enumerate(batch_generator):
            # Compute the output
            logits = self(x=batch_dict['x_data'])

            # Compute the loss
            target = self._prepare_lossfunc(batch_dict['y_target'])
            loss = self._criterion(logits, target)
            batch_loss = loss.to("cpu").item()
            running_loss += (batch_loss - running_loss) / (batch_index + 1)
            
            # Compute the accuracy
            batch_acc = self._compute_accuracy(logits, target)
            running_acc += (batch_acc - running_acc) / (batch_index + 1)

            # Compute F1-score
            batch_f1 = self._compute_f1(logits, target)
            running_f1 += (batch_f1 - running_f1) / (batch_index + 1)

        return running_loss, running_acc, running_f1

    def train_net(self):
        # Initiate training mode
        self.train()

        self._args.dataset.set_split('train')
        batch_generator = self._args.dataset.generate_batches (
            batch_size=self._args.batch_size, 
            device=self._args.device
        )

        running_loss = 0
        running_acc = 0
        running_f1 = 0

        for batch_index, batch_dict in enumerate(batch_generator):
            # Zero gradients
            self._optimizer.zero_grad()

            # Perform a forward pass
            logits = self(x=batch_dict['x_data'])

            # Compute the loss for that pass
            target = self._prepare_lossfunc(batch_dict['y_target'])
            loss = self._criterion(logits, target)
            batch_loss = loss.to("cpu").item()
            running_loss += (batch_loss - running_loss) / (batch_index + 1)
            
            # Use computed loss to produce gradients
            loss.backward()

            # Use the optimizer to take gradient step
            self._optimizer.step()

            batch_acc = self._compute_accuracy(logits, target)
            running_acc += (batch_acc - running_acc) / (batch_index + 1)

            batch_f1 = self._compute_f1(logits, target)
            running_f1 += (batch_f1 - running_f1) / (batch_index + 1)

        return running_loss, running_acc, running_f1

    def plot_logs(self, title, legend):
        plt.figure(figsize=(10, 5))
        plt.title(title)

        epochs = [(e + 1) for e in range(self._args.num_epochs)]

        if title == "Accuracy":
            train_metric = "train_acc"
            val_metric = "val_acc"
        elif title == "Loss":
            train_metric = "train_loss"
            val_metric = "val_loss"
        else:
            train_metric = "train_f1"
            val_metric = "val_f1"

        plt.plot(epochs, self._logs[train_metric])
        plt.plot(epochs, self._logs[val_metric])

        plt.legend(legend)
        plt.show()

class BOWClassifier(MLP):
    def __init__(self, args):
        super().__init__(args)

        self._topology = nn.Sequential (
            nn.Linear(self._args.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self._args.out_units)
        )

    def forward(self, x):
        for layer in self._topology:
            x = layer(x)
        return x
        
class CNNClassifier(MLP):
    def __init__(self, args):
        super().__init__(args)

        self._embedding = nn.Embedding.from_pretrained (
            args.pretrained_embedding, 
            freeze=args.freeze_embedding
        )

        # Conv Network
        self._conv1d_list = nn.ModuleList ([
            nn.Conv1d(
                in_channels=args.embed_dim,
                out_channels=args.num_filters[i],
                kernel_size=args.filter_sizes[i]
            ) for i in range(len(args.filter_sizes))
        ])

        # Fully-connected layer and Dropout
        self._fc = nn.Linear(np.sum(args.num_filters), args.out_units)
        self._dropout = nn.Dropout(p=0.5)

    @property
    def embedding(self):
        return self._embedding

    @property
    def conv1d_list(self):
        return self._conv1d_list

    @property
    def fc(self):
        return self._fc
    
    @property
    def dropout(self):
        return self._dropout

    def forward(self, x):
        """ Perform a forward pass through the network.

        Args:
            x (torch.Tensor): A tensor of token ids with shape (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size, n_classes)
        """

        # Get embeddings from `input_ids`. 
        # Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(x).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
        
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
        
        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits

class LSTMClassifier(MLP):
    def __init__(self, args):
        super().__init__(args)

        self._embedding = nn.Embedding.from_pretrained (
            args.pretrained_embedding, 
            freeze=args.freeze_embedding
        )

        # Setup LSTM
        self._lstm = nn.LSTM (
            input_size=args.embed_dim, 
            hidden_size=args.hidden_size,
            num_layers=args.num_layers, 
            batch_first=True
        ) 
        
        self._fc_1 = nn.Linear(args.hidden_size, 128) 
        self._fc_2 = nn.Linear(128, args.out_units)

        self._relu = nn.ReLU()

    @property
    def embedding(self):
        return self._embedding  

    @property
    def lstm(self):
        return self._lstm

    @property
    def fc_1(self):
        return self._fc_1

    @property
    def relu(self):
        return self._relu

    @property
    def fc_2(self):
        return self._fc_2
    
    def forward(self, x):
        # Use pretrained embeddings
        x = self.embedding(x).float()
        
        # Calculate initial hidden & internal states
        h_0 = torch.zeros(self._args.num_layers, x.size(0), self._args.hidden_size).to(self._args.device)
        c_0 = torch.zeros(self._args.num_layers, x.size(0), self._args.hidden_size).to(self._args.device) 

        # Propagate input through LSTM
        out, _ = self.lstm(x, (h_0, c_0)) 

        # Reshape input
        out = out[:, -1, :]

        # Forward input through 
        # the fully connected layers
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)

        return out