import math
import torch
from torch.utils.data.dataloader import DataLoader
import pdb

from .timer import Timer

def to(xs, device):
    return [x.to(device) for x in xs]

class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.n_epochs = 0
        self.n_tokens = 0 # counter used for learning rate decay
        self.optimizer = None

    def get_optimizer(self, model):
        if self.optimizer is None:
            print(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}')
            self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def train(self, model, dataset, n_epochs=1, log_freq=100):

        config = self.config
        optimizer = self.get_optimizer(model)
        model.train(True)
        vocab_size = dataset.N

        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        for _ in range(n_epochs):

            losses = []
            timer = Timer()
            for it, batch in enumerate(loader):

                batch = to(batch, self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    logits, loss = model(*batch)
                    losses.append(loss.item())

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    y = batch[-2]
                    self.n_tokens += (y != vocab_size).sum() # number of tokens processed this step
                    if self.n_tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.n_tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # report progress
                if it % log_freq == 0:
                    print(
                        f'[ utils/training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                        f'train loss {loss.item():.5f} | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                        f't: {timer():.2f}')

            self.n_epochs += 1
