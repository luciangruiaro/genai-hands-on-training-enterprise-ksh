import os
import random
import json
from dataclasses import dataclass
from typing import List
from config import global_config as cfg


@dataclass
class TrainingData:
    input: List[float]
    output: List[float]


def one_hot(label: int) -> List[float]:
    return [1.0 if i == label else 0.0 for i in range(10)]


def get_training_data() -> List[TrainingData]:
    resolution = cfg.APP_CONFIG["nn"]["default_resolution"]
    return load_or_generate_dataset(resolution)


def get_default_inference_sample() -> TrainingData:
    resolution = cfg.APP_CONFIG["nn"]["default_resolution"]
    digit = cfg.APP_CONFIG["nn"].get("default_digit", 2)

    path = f"resources/mnist_samples/{resolution}x{resolution}/digit_{digit}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            obj = json.load(f)
            return TrainingData(input=obj["input"], output=obj["output"])

    dataset = load_or_generate_dataset(resolution)
    samples = [sample for sample in dataset if sample.output.index(1.0) == digit]
    if not samples:
        raise ValueError(f"No samples found for digit {digit} at resolution {resolution}x{resolution}")

    selected = random.choice(samples)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(selected.__dict__, f, indent=2)
    return selected


def load_or_generate_dataset(resolution: int) -> List[TrainingData]:
    path = f"resources/training_data/{resolution}x{resolution}/full_dataset.json"

    if os.path.exists(path):
        with open(path, "r") as f:
            raw = json.load(f)
            return [TrainingData(**entry) for entry in raw]

    data = generate_dataset_for_resolution(resolution)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([entry.__dict__ for entry in data], f, indent=2)

    return data


def generate_dataset_for_resolution(resolution: int) -> List[TrainingData]:
    if resolution == 8:
        return load_digits_dataset()
    elif resolution == 14:
        return downsample_mnist_dataset(14)
    elif resolution == 28:
        return load_mnist_dataset()
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


def load_digits_dataset() -> List[TrainingData]:
    from sklearn.datasets import load_digits
    data = load_digits()
    return [TrainingData(input=img.tolist(), output=one_hot(label)) for img, label in zip(data.data, data.target)]


def load_mnist_dataset() -> List[TrainingData]:
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    return [TrainingData(input=img.flatten().tolist(), output=one_hot(label)) for img, label in zip(x_train, y_train)]


def downsample_mnist_dataset(resolution: int) -> List[TrainingData]:
    from tensorflow.keras.datasets import mnist
    from skimage.transform import resize
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0

    return [
        TrainingData(
            input=resize(img, (resolution, resolution), anti_aliasing=True).flatten().tolist(),
            output=one_hot(label)
        ) for img, label in zip(x_train, y_train)
    ]
