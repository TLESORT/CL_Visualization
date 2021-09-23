
import torch

def get_Output_layer(LayerName, in_dim, out_dim):
    if LayerName == "CosLayer":
        from Models.Output_Layers.layer import CosineLayer
        # We replace the output layer by a cosine layer
        outlayer = CosineLayer(in_dim, out_dim)
    elif LayerName == "WeightNorm":
        #from Models.Output_Layers.layer import WeightNormLayer
        #outlayer = WeightNormLayer(in_dim, out_dim, bias=False)

        from torch.nn.utils import weight_norm
        outlayer = weight_norm(torch.nn.Linear(in_dim, out_dim, bias=False))
        outlayer.weight_g.requires_grad = False # remove g parameter from the parametrization
        outlayer.state_dict()['weight_g'] = torch.ones(out_dim)

    elif LayerName == "OriginalWeightNorm":
        from torch.nn.utils import weight_norm
        outlayer = weight_norm(torch.nn.Linear(in_dim, out_dim, bias=False))
    elif LayerName == "SLDA":
        from Models.Output_Layers.layer import SLDALayer
        outlayer = SLDALayer(in_dim, out_dim)
    elif LayerName == "Linear":
        outlayer = torch.nn.Linear(in_dim, out_dim, bias=True)
    elif LayerName == "Linear_no_bias":
        outlayer = torch.nn.Linear(in_dim, out_dim, bias=False)
    elif "MIMO_" in LayerName:
        from Models.Output_Layers.layer import MIMO
        outlayer = MIMO(in_dim, out_dim, num_layer=3, layer_type=LayerName)
    elif LayerName == "MeanLayer":
        from Models.Output_Layers.layer import MeanLayer
        outlayer = MeanLayer(in_dim, out_dim)
    elif LayerName == "MedianLayer":
        from Models.Output_Layers.layer import MedianLayer
        outlayer = MedianLayer(in_dim, out_dim)
    elif LayerName == "KNN":
        from Models.Output_Layers.layer import KNN
        outlayer = KNN(in_dim, out_dim, K=5)
    else:
        raise NotImplementedError
    return outlayer

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model