from __future__ import print_function
import datetime
import os
import toml
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.cuda as cuda
from models import LipRead
from training import Trainer
from validation import Validator
from checkpoint import *
from csv_saver import CSVSaver


def load_pretrained(model: nn.Module, pretrained_model_filepath: str, freeze: bool):
    if cuda.is_available():
        pretrained_dict = torch.load(pretrained_model_filepath)
    else:
        pretrained_dict = torch.load(pretrained_model_filepath, map_location=lambda storage, loc: storage)

    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    if freeze:
        # free already trained parameters
        for name, param in model.named_parameters():
            if name in pretrained_dict.keys():
                param.requires_grad_(False)


def create_run_name() -> str:
    return datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')


def create_result_dir(run_name: str, root_dir=None) -> str:
    """Create result dir and return its absolute path
    """
    if root_dir is None:
        root_dir = os.path.dirname(os.path.realpath(__file__))
    
    result_dir = os.path.join(root_dir, run_name)
    if os.path.exists(result_dir):
        raise Exception("directory {} already exists".format(result_dir))
    os.makedirs(result_dir)
    return result_dir


if __name__ == '__main__':
    parser = ArgumentParser(description="train LipReading-ResNet")
    parser.add_argument("--run_name", dest="run_name", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_file", default=None)
    parser.add_argument("--final_epoch", dest="final_epoch", default=None)
    parser.add_argument("--root_dir", dest="root_dir", default=None)
    args = parser.parse_args()

    # for saving training metrics
    run_name = create_run_name() if args.run_name is None else args.run_name
    result_dir = create_result_dir(run_name, args.root_dir)
    csv = CSVSaver(os.path.join(result_dir, "stats.csv"), "accuracy", "loss")

    if args.checkpoint_file is None:
        print("Loading options...")
        with open('options.toml', 'r') as optionsFile:
            options = toml.loads(optionsFile.read())
        model = LipRead(options)
        last_epoch = options["training"]["startepoch"] - 1

        # specify pretrained frontend model when training backend
        if options["general"]["train_target"] == 'backend':
            load_pretrained(model, options["general"]["frontend_pretrained_model_path"], freeze=True)

    else:
        state_dict, grad_states, states = load_checkpoint(args.checkpoint_file)
        options = states['options']
        model = LipRead(options)
        load_model(model, state_dict, grad_states) # load weights and freeze states
        last_epoch = states["epoch"]

    
    # save current options
    with open(os.path.join(result_dir, "options_used.toml"), 'w') as f:
        toml.dump(options, f)


    if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True


    if options["training"]["train"]:
        trainer = Trainer(options)
    if(options["validation"]["validate"]):
        validator = Validator(options, result_dir)


    final_epoch = args.final_epoch if args.final_epoch is not None else options["training"]["epochs"]
    for epoch in range(last_epoch + 1, final_epoch):
        loss = trainer.epoch(model, epoch) if options["training"]["train"] else ''
        accuracy = validator.epoch(model, epoch) if options["validation"]["validate"] else ''
        csv.add(epoch, accuracy=accuracy, loss=loss)
        save_checkpoint(result_dir, epoch, model, options=options)

    # save the final model 
    torch.save(model.state_dict(), os.path.join(result_dir, "epoch{}.pt".format(epoch)))
