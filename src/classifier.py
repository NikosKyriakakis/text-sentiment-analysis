import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
    

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self._topology = None
        self._args = args
        self._optimizer = None
        self._criterion = None
        self._logs = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

    def setup(self):
        if self._args.binary_class == True:
            self._out_activation = torch.sigmoid
        else:
            self._out_activation = torch.softmax
        self.configure_optimizer(self._args.optimizer)
        self.configure_loss_function(self._args.criterion)

    def configure_loss_function(self, criterion):
        if criterion == "bce_logits":
            self._criterion = nn.BCEWithLogitsLoss()

    def configure_optimizer(self, optimizer):
        if optimizer == "Adam":
            self._optimizer = optim.Adam(self.parameters(), lr=self._args.learning_rate)

    def forward(self, X):
        pass

    def compute_accuracy(self, y_pred, y_target):
        y_target = y_target.cpu()
        y_pred_indices = (self._out_activation(y_pred) > 0.5).cpu().long()
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    def checkpoint(self):
        size = len(self._logs["val_loss"])
        if size > 1:
            if self._logs["val_loss"][size - 1] < self._logs["val_loss"][size - 2]:
                torch.save(self.state_dict(), self._args.save_dir)

    def fit(self):
        for _ in tqdm(range(self._args.num_epochs)):
            self.checkpoint()

            train_loss, train_acc = self.train_net()
            self._logs["train_loss"].append(train_loss)
            self._logs["train_acc"].append(train_acc)

            val_loss, val_acc = 0, 0
            with torch.no_grad():
                val_loss, val_acc = self.eval_net(mode='val')
                self._logs["val_loss"].append(val_loss)
                self._logs["val_acc"].append(val_acc)
        
    def eval_net(self, mode):
        self.eval()

        self._args.dataset.set_split(mode)
        batch_generator = self._args.dataset.generate_batches (
            batch_size=self._args.batch_size, 
            device=self._args.device
        )

        running_loss = 0
        running_acc = 0

        for batch_index, batch_dict in enumerate(batch_generator):
            # Compute the output
            logits = self(x=batch_dict['x_data'].float())

            # Compute the loss
            loss = self._criterion(logits, batch_dict['y_target'].float())
            batch_loss = loss.to("cpu").item()
            running_loss += (batch_loss - running_loss) / (batch_index + 1)
            
            # Compute the accuracy
            batch_acc = self.compute_accuracy(logits, batch_dict['y_target'].float())
            running_acc += (batch_acc - running_acc) / (batch_index + 1)

        return running_loss, running_acc


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

        for batch_index, batch_dict in enumerate(batch_generator):
            # Zero gradients
            self._optimizer.zero_grad()

            # Perform a forward pass
            logits = self(x=batch_dict['x_data'].float())

            # Compute the loss for that pass
            loss = self._criterion(logits, batch_dict['y_target'].float())
            batch_loss = loss.to("cpu").item()
            running_loss += (batch_loss - running_loss) / (batch_index + 1)
            
            # Use computed loss to produce gradients
            loss.backward()

            # Use the optimizer to take gradient step
            self._optimizer.step()

            batch_acc = self.compute_accuracy(logits, batch_dict['y_target'].float())
            running_acc += (batch_acc - running_acc) / (batch_index + 1)

        return running_loss, running_acc

    def plot_loss_logs(self):
        plt.figure(figsize=(10, 5))
        plt.title("Loss")
        epochs = [e for e in range(self._args.num_epochs)]
        plt.plot(epochs, self._logs["train_loss"])
        plt.plot(epochs, self._logs["val_loss"])
        plt.legend(['Train-loss', 'Validation-loss'])
        plt.show()

    def plot_acc_logs(self):
        plt.figure(figsize=(10, 5))
        plt.title("Accuracy")
        epochs = [e for e in range(self._args.num_epochs)]
        plt.plot(epochs, self._logs["train_acc"])
        plt.plot(epochs, self._logs["val_acc"])
        plt.legend(['Train-accuracy', 'Validation-accuracy'])
        plt.show()

class BOWClassifier(MLP):
    def __init__(self, args):
        super().__init__(args)

        self._topology = nn.Sequential (
            nn.Linear(self._args.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        for layer in self._topology:
            x = layer(x)
        return x.squeeze()
        