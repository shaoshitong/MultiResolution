import math


def grad_print(model):
    for name, parameter in model.named_parameters():
        if type(parameter.grad) != type(None):
            print(name, math.exp(parameter.grad.data.abs().mean().item()-1))