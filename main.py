from __future__ import print_function
import datetime
import os
from models import LipRead
import torch
import torch.nn as nn
import toml
from training import Trainer
from validation import Validator


def load_pretrained(model: nn.Module, pretrained_model_filepath: str, freeze: bool):
    # load pretrained model
    pretrained_dict = torch.load(pretrained_model_filepath)
    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    if freeze:
        # free already trained parameters
        for name, param in model.named_parameters():
            if name in pretrained_dict.keys():
                param.require_grad_(False)


run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
current_dir = os.path.dirname(os.path.real_path(__file__))
result_dir = os.path.join(current_dir, "results", run_name)
model_dir = os.path.join(current_dir, "models", run_name)
os.makedir(result_dir)
os.makedir(model_dir)


print("Loading options...")
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

#Create the model.
model = LipRead(options)

# switch pretreained model loading
if options["general"]["train_target"] == 'frontend':
    if options["general"]["load_pretrained_model"]:
        load_pretrained(model, options["general"]["frontend_pretrained_model_path"]), false)
if options["general"]["train_target"] == 'backend':
    load_pretrained(model, options["general"]["frontend_pretrained_modelpath"]), true)
    if(options["general"]["load_pretrained_model"]):
        load_pretrained(model, options["general"]["backend_pretrained_model_path"]), false)


if options["training"]["train"]:
    trainer = Trainer(options, model_dict)

if(options["validation"]["validate"]):
    validator = Validator(options, result_dir)

for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):

    if(options["training"]["train"]):
        trainer.epoch(model, epoch)

    if(options["validation"]["validate"]):
        validator.epoch(model, epoch)
