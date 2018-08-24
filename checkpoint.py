import os
import torch
import torch.nn as nn
import torch.cuda as cuda


def save_checkpoint(save_dir_path: str, epoch: int, model: nn.Module, **kwargs):
    # save freeze state
    grad_states = {}
    for n, p in model.named_parameters():
        grad_states[n] = p.requires_grad

    states = {'state_dict': model.state_dict(), 'grad_states': grad_states, 'epoch': epoch}
    states.update(kwargs)
    torch.save(state, os.path.join(save_dir_path, "checkpoint_ep{}.pt".format(epoch)))


def load_checkpoint(filename: str) -> tuple:
    """Return (weights, grad_states, states)
    Load weights using torch.load_state_dict.
    grad_states is a dictionary where requires_grad is stored for each parameter.
    states contains such as accuracy and loss depending on what you save in save_checkpoint.
    """
    if cuda.is_available():
        states = torch.load(filename)
    else:
        states = torch.load(filename, map_locatioin=lambda storage, loc: storage)
    return states.pop('state_dict'), states.pop('grad_states'), states
    

def load_model(model: nn.Module, state_dict: dict, grad_states: dict):
    """Load model parameters and parameter freeze states
    """
    model.load_state_dict(state_dict)
    for n, p in model.named_parameters():
        p.requires_grad_(grad_states[n])