import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from main import CustomDataset
from model import MyModel


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        bs = x.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def log_images(writer: SummaryWriter, model, dataset, device, epoch: int, k: int = 5):
    model.eval()
    imgs = []
    for i in range(k):
        x, y = dataset[i]
        x = x.unsqueeze(0).to(device)
        pred = model(x).cpu()
        gt = y.unsqueeze(0).cpu()
        imgs.append(gt.squeeze(0))
        imgs.append(pred.squeeze(0))
    grid = vutils.make_grid(imgs, nrow=2)
    writer.add_image("GT_vs_Pred", grid, epoch)


def train(
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        epochs: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 12,
        min_delta: float = 1e-5,
        run_dir: str = "runs/model",
        weights_dir: str = "weights",
):
    set_seed(42)
    device = torch.device(device)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)

    ds = CustomDataset(train=True, cache=True)
    n_total = len(ds)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type != "cpu"))

    model = MyModel(width=32).to(device)
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.SmoothL1Loss(beta=0.02)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # opt, mode="min", factor=0.5, patience=3, min_lr=1e-5)

    es = EarlyStopping(patience=patience, min_delta=min_delta)
    best_path = os.path.join(weights_dir, "model_best.pt")

    start_time = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            bs = x.size(0)
            running += loss.item() * bs
            n += bs

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            global_step += 1

        train_loss = running / max(1, n)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        # scheduler.step(val_loss)

        # current_lr = opt.param_groups[0]["lr"]
        # writer.add_scalar("train/lr", current_lr, epoch)
        # if epoch % 5 == 0:
        #     print(f"   lr={current_lr:.6e}")

        writer.add_scalar("train/loss_epoch", train_loss, epoch)
        writer.add_scalar("val/loss_epoch", val_loss, epoch)

        if epoch == 1 or epoch % 5 == 0:
            log_images(writer, model, ds, device, epoch, k=5)

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss <= es.best + 1e-12:
            torch.save({"model_state": model.state_dict()}, best_path)

        if es.step(val_loss):
            print(f"Early stopping la epoch {epoch}. Best val_loss={es.best:.6f}")
            break

    writer.add_scalar("train/total_time_sec", time.time() - start_time, 0)
    writer.close()
    print(f" Weights salvate: {best_path}")
    return best_path

if __name__ == "__main__":
    dev = "cpu"
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"

    train(device=dev)