
import torch

def get_Output_layer(LayerName, in_dim, out_dim):
    if LayerName == "CosLayer":
        from Models.Output_Layers.layer import CosineLayer
        # We replace the output layer by a cosine layer
        outlayer = CosineLayer(in_dim, out_dim)
    elif LayerName == "SLDA":
        from Models.Output_Layers.layer import SLDALayer
        # We replace the output layer by a cosine layer
        outlayer = SLDALayer(in_dim, out_dim)
    elif LayerName == "Linear_no_bias":
        outlayer = torch.nn.Linear(in_dim, out_dim, bias=False)
    else:
        outlayer = torch.nn.Linear(in_dim, out_dim, bias=True)
    return outlayer

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model