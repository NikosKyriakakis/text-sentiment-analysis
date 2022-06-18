import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score

plt.style.use('ggplot')


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self._topology = None
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
        self.__configure_loss_function(self._args.criterion)
        self.__configure_optimizer(self._args.optimizer)
        self.__configure_metrics_method(self._args.binary_class)

    def __configure_loss_function(self, criterion):
        if criterion == "bce_logits":
            self._criterion = nn.BCEWithLogitsLoss()
        else:
            self._criterion = nn.CrossEntropyLoss()

    def __configure_optimizer(self, optimizer):
        if optimizer == "Adam":
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

    def forward(self, x):
        for layer in self._topology:
            x = layer(x)
        return x

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

    def checkpoint(self):
        size = len(self._logs["val_loss"])
        if size > 1:
            if self._logs["val_loss"][size - 1] < self._logs["val_loss"][size - 2]:
                torch.save(self.state_dict(), self._args.save_dir)

    def _prepare_bin_lossfunc(self, tensor_data):
        return tensor_data.view(-1, 1).float()

    def _prepare_mult_lossfunc(self, tensor_data):
        return tensor_data.long()

    def fit(self):
        # Send model to available hardware
        self = self.to(self._args.device)

        for _ in tqdm(range(self._args.num_epochs)):
            # self.checkpoint()

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
            logits = self(x=batch_dict['x_data'].float())

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
            logits = self(x=batch_dict['x_data'].float())

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
        
# class CNNClassifier(MLP):
#     def __init__(self, args):
#         super().__init__(args)

#         self._topology = nn.Sequential (
#             nn.Conv1d(in_channels=self._args.in_features, out_channels=),
#         )

#         self.convs = nn.ModuleList([
#             nn.Conv2d (
#                 in_channels=1, 
#                 out_channels=n_filters, 
#                 kernel_size=(fs, embedding_dim)
#             ) for fs in filter_sizes
#         ])
        
#         self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
#         self.dropout = nn.Dropout(dropout)

# class CNN(MLP):
#     def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
#                  dropout, pad_idx):
        
#         super().__init__()
                
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
#         self.convs = nn.ModuleList([
#                                     nn.Conv2d(in_channels = 1, 
#                                               out_channels = n_filters, 
#                                               kernel_size = (fs, embedding_dim)) 
#                                     for fs in filter_sizes
#                                     ])
        
#         self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
#         self.dropout = nn.Dropout(dropout)