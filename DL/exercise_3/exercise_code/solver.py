import numpy as np
import torch
from torch.autograd import Variable



class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        #######################################################################
        # TODO:                                                               #
        # Write your own personal training method for our solver. In each     #
        # epoch iter_per_epoch shuffled training batches are processed. The   #
        # loss for each batch is stored in self.train_loss_history. Every     #
        # log_nth iteration the loss is logged. After one epoch the training  #
        # accuracy of the last mini batch is logged and stored in             #
        # self.train_acc_history. We validate at the end of each epoch, log   #
        # the result and store the accuracy of the entire validation set in   #
        # self.val_acc_history.                                               #
        #                                                                     #
        # Your logging could like something like:                             #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                           #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                           #
        #   ...                                                               #
        #######################################################################
        for epoch in range(num_epochs):
            #set up the model for training
            model.train()
            it = 0

            for batch in train_loader:
                # Set up the batch and the model
                inputs, targets = batch
                inputs, targets = Variable(inputs), Variable(targets)
                
                optim.zero_grad()
                
                # forward step
                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)
                
                #backward step
                loss.backward()
                optim.step()
                
                
                loss_val = loss.item()
                self.train_loss_history.append(loss_val)
                it += 1
                if (it % log_nth ==0):
                    print("[Iteration %d/%d] TRAIN loss: %.3f" % (it, iter_per_epoch, loss_val))
                    
            # Training Accuracy on the last mini batch
            _, preds = torch.max(outputs, 1)
            targets_idxs = targets >= 0
            train_acc = np.mean((preds == targets)[targets_idxs].data.numpy())
            self.train_acc_history.append(train_acc)
            print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, train_acc, loss_val))
        
            model.eval()
                                
            val_accs = []
            val_losses = []
            
            for batch in val_loader:
                # Set up the batch and the model
                inputs, targets = batch
                inputs, targets = Variable(inputs), Variable(targets)

                optim.zero_grad()
                
                # forward step
                outputs = model(inputs)
                loss = self.loss_func(outputs, targets)

                loss_val = loss.item()
                self.val_loss_history.append(loss_val)
                
                _, preds = torch.max(outputs, 1)
                targets_idxs = targets >= 0
                val_acc = np.mean((preds == targets)[targets_idxs].data.numpy())
                val_accs.append(val_acc)
                val_losses.append(loss_val)
                                  
            val_acc_mean = np.mean(val_accs)
            val_loss_mean = np.mean(val_losses)
            print('[Epoch %d/%d] VAL acc/loss: %.3f/%.3f' % (epoch + 1, num_epochs, val_acc_mean, val_loss_mean))
                                
            # Validation acc on the entire val dataset
                
        
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
