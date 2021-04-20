
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
    elif LayerName == "MIMO":
        from Models.Output_Layers.layer import MIMO
        outlayer = MIMO(in_dim, out_dim, num_layer=3, layer_type="Linear")
    elif LayerName == "MeanLayer":
        from Models.Output_Layers.layer import MeanLayer
        outlayer = MeanLayer(in_dim, out_dim)
    elif LayerName == "KNN":
        from Models.Output_Layers.layer import KNN
        outlayer = KNN(in_dim, out_dim, K=5)
    else:
        outlayer = torch.nn.Linear(in_dim, out_dim, bias=True)
    return outlayer

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model