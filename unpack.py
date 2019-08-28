import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--output", type=str, default="")
args = parser.parse_args()


checkpoint = torch.load(args.checkpoint, map_location="cpu")
torch.save(checkpoint["model"], os.path.join(args.output, "pytorch_model.bin"))
