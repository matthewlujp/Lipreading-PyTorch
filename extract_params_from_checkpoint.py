import os
import sys
import shutil
from argparse import ArgumentParser
import toml
import torch
from models import LipRead
from checkpoint import *


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("checkpoint_file", help="path to checkpoint file")
    parser.add_argument("save_dir", help="path to save extracted data")
    args = parser.parse_args()

    state_dict, grad_states, states = load_checkpoint(args.checkpoint_file)

    # prepare directory to save
    save_dir = args.save_dir
    if os.path.exists(save_dir):
        res = input("Directory {} exists. Are you sure to remove {}? -- y/n".format(save_dir))
        if res == 'y':
            shutil.rmtree(save_dir)
        else:
            sys.exit(-1)
    os.makedirs(save_dir)

    options = states['options']
    with open(os.path.join(save_dir, "options.toml"), 'w') as f:
        toml.dump(options, f)

    model = LipRead(options)
    load_model(model, state_dict, grad_states)
    epoch = states['epoch']
    torch.save(model.state_dict(), os.path.join(save_dir, "model_{}ep.pt".format(epoch)))

    
    