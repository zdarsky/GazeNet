import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import copy
from tqdm import tqdm
import glob


class GazeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(14, 200)
        self.sig = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(200, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sig(x)
        x = self.linear2(x)
        return x


def dist_loss(_input, target):
    loss = ((_input - target) ** 2).sum(1)
    return (loss ** 0.5).sum() / loss.data.nelement()


class NetWrapper():
    run_id = -1

    model = GazeNet()
    opt = None
    scheduler = None

    epochs = None
    epoch = 0
    testing = True
    resume = True
    writer = None
    step = 10

    dataloader = None
    train_x = None
    train_y = None
    valid_x = None
    valid_y = None

    train_loss_min = np.inf
    valid_loss_min = np.inf

    best_model = model  # deepcopy init?
    folder = "/Users/nzdarsky/code/thesis_bachelor/data"

    def __init__(self, run_id, testing=True, resume=True):
        self.run_id = run_id
        self.testing = testing
        self.resume = resume
        # TODO: fix path
        if not testing:
            self.writer = SummaryWriter(f"{self.folder}/writer/{run_id}")

    def init(self, epochs, learning_rate, momentum):
        self.epochs = epochs
        self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                   momentum=momentum)
        if self.resume:
            ckpts = glob.glob(f"{self.folder}/runs/{self.run_id}_ckpt_*.pt")
            if ckpts:
                # load latest checkpoint
                print(f"trying to load: {ckpts[-1]}")
                state = torch.load(ckpts[-1])
                self.epoch = state["epoch"]
                if self.epoch >= self.epochs:
                    # exit("Error: already reached final epoch, aborting...")
                    print("Already at final epoch")
                self.scheduler = state["scheduler"]
                self.model.load_state_dict(state["model"])
                self.opt = torch.optim.SGD(self.model.parameters(),
                                           lr=learning_rate,
                                           momentum=momentum)
                self.opt.load_state_dict(state["optimizer"])
        return self

    def set_ds(self, train_x, train_y, valid_x, valid_y):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        return self

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader
        return self

    def set_scheduler_inteval(self, sched_interval):
        # could have just used StepLR...
        milestones = list(range(sched_interval, self.epochs, sched_interval))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt,
                                                              milestones, 0.5)
        return self

    def make_train_step(self):
        self.model.train()
        for x_batch, y_batch in self.dataloader:
            self.opt.zero_grad()
            output = self.model(x_batch)
            train_loss = dist_loss(output, y_batch)
            train_loss.backward()
            self.opt.step()
        self.scheduler.step()  # only updates every 10 epochs atm?

    # TODO: use dataloader?
    def make_valid_step(self):
        train_loss = dist_loss(self.model(self.train_x), self.train_y)
        if train_loss < self.train_loss_min:
            self.train_loss_min = train_loss

        if not self.testing:
            self.writer.add_scalar("train_loss", train_loss.item(), self.epoch)

        self.model.eval()
        valid_loss = dist_loss(self.model(self.valid_x), self.valid_y)
        if valid_loss < self.valid_loss_min:
            self.valid_loss_min = valid_loss
            self.best_model = copy.deepcopy(self.model)

        if not self.testing:
            self.writer.add_scalar("valid_loss", valid_loss.item(), self.epoch)
        return train_loss, valid_loss

    # TODO: improve dl & ds stuff
    def fit(self):
        valid_loss = np.inf
        train_loss = np.inf

        for self.epoch in tqdm(range(self.epochs)):
            self.make_train_step()
            if self.epoch % self.step == 0:
                train_loss, valid_loss = self.make_valid_step()

        print(f"training finished after {self.epochs} epochs, "
              f"lowest train_loss: {self.train_loss_min:.3f}, "
              f"lowest valid_loss: {self.valid_loss_min:.3f}")

    def test(self, eval_x, eval_y):
        self.best_model.eval()
        print(f"test_err: {int(dist_loss(self.best_model(eval_x), eval_y))}")

    def eval(self, eval_x):
        self.best_model.eval()
        return self.best_model(eval_x)

    def save_best_model(self):
        state = {
            "epoch": self.epoch + 1,
            "arch": str(self.best_model),
            "model": self.best_model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.scheduler
        }
        torch.save(state,
                   f"{self.folder}/runs/{self.run_id}_ckpt_{self.epoch+1}.pt")
