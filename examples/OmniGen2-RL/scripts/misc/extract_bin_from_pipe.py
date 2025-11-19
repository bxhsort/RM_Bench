import dotenv

dotenv.load_dotenv(override=True)

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel


def main():
    transformer = OmniGen2Transformer2DModel.from_pretrained("/mnt/Reward/EditScore/examples/OmniGen2-RL/OmniGen2/models--OmniGen2--OmniGen2/snapshots/df5dca8a981d74e6c3af214c145f5c735fe72367", subfolder="transformer")

    state_dict = transformer.state_dict()

    save_path = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "pretrained_models", "OmniGen2", "transformer/pytorch_model.bin")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving model to {save_path}")
    torch.save(state_dict, save_path)

if __name__ == "__main__":
    main()
