import torch
from selfatt import device


class TrainingAssistant:
    def __init__(self, batch_size, block_size, eval_iterations):
        self.batch_size = batch_size
        self.block_size = block_size
        self.eval_iterations = eval_iterations

    def get_batch(self, data):
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model, data):
        model.eval()
        losses = torch.zeros(self.eval_iterations)
        for k in range(self.eval_iterations):
            X, Y = self.get_batch(data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        model.train()
        return losses.mean()
