from sklearn.metrics import accuracy_score
import torch

class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass

class Accuracy(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self):
        self.n = 0
        self.name = 'Accuracy'
        self.match = 0

    def reset(self):
        self.n = 0
        self.match = 0
        
    def update(self, predicts, label):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts, label = predicts.cpu(), label.cpu()
        batch_size = list(predicts.size())[0]
        _, y_pred = torch.max(predicts, dim=1)
        self.match += accuracy_score(label, y_pred, normalize=False)
        self.n += batch_size
    
    def print_score(self):
        acc = self.match / self.n
        return '{:.4f}'.format(acc)
