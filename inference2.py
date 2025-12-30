import torch
from model2 import MyModel
from main import test_inference_time


def load_model(weights_path: str, device: str):
    device_t = torch.device(device)
    model = MyModel(width=48).to(device_t)
    ckpt = torch.load(weights_path, map_location=device_t)
    model.load_state_dict(ckpt["model_state"])
    return model


if __name__ == "__main__":
    weights = "weights/model_best.pt"

    model_cpu = load_model(weights, "cpu")
    test_inference_time(model_cpu, device=torch.device("cpu"))


    if torch.backends.mps.is_available():
        model_mps = load_model(weights, "mps")
        test_inference_time(model_mps, device=torch.device("mps"))
    elif torch.cuda.is_available():
        model_cuda = load_model(weights, "cuda")
        test_inference_time(model_cuda, device=torch.device("cuda"))
