import os
import torch
import matplotlib.pyplot as plt

from main import CustomDataset
from model2 import MyModel


def save_examples(weights_path="weights/model_best.pt", device="cpu", out_dir="examples", k=5):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    ds = CustomDataset(train=False, cache=False)

    model = MyModel(width=48).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        for i in range(k):
            x, y_true = ds[i]
            x_b = x.unsqueeze(0).to(device)
            y_pred = model(x_b).squeeze(0).cpu()

            fig = plt.figure(figsize=(9, 3))

            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(x.permute(1, 2, 0).cpu())
            ax1.set_title("Input")
            ax1.axis("off")

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(y_true.squeeze(0).cpu(), cmap="gray")
            ax2.set_title("Ground truth")
            ax2.axis("off")

            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(y_pred.squeeze(0), cmap="gray")
            ax3.set_title("Prediction")
            ax3.axis("off")

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"ex_{i}.png"), dpi=200)
            plt.close(fig)

    print(f" Exemple salvate în: {out_dir}/ex_0.png ... ex_{k-1}.png")


if __name__ == "__main__":
    dev = "cpu"
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"

    save_examples(device=dev)
