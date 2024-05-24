import torch

class EarlyStopping(object):
    def __init__(self, patience=3, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience  # Number of epochs to wait for improvement
        self.delta = delta  # Minimum change in the monitored metric to be considered as an improvement
        self.verbose = verbose  # Whether to print information about the early stopping process
        self.path = path  # Path to save the model checkpoint
        self.counter = 0  # Counter to track the number of epochs without improvement
        self.best_score = None  # Best validation score (metric)
        self.best_epoch = None
        self.early_stop = False  # Whether to stop training early

    def __call__(self, val_loss, epoch, model=None):
        # If the best_score is None (first epoch), initialize it with the current validation loss
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = epoch
            # self.save_checkpoint(val_loss, model)
        # If the current validation loss is worse than the best_score - delta, increment the counter
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'At epoch {epoch}, EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                print(f'At epoch {epoch}, the valid errors has not been decreased within 3 epochs, thus ending training')
                self.early_stop = True
        # If the current validation loss is better than the best_score - delta, update the best_score
        else:
            self.best_score = val_loss
            self.best_epoch = epoch
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)