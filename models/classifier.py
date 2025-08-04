import torch.nn as nn
import timm

def build_model(model_name = "mobilenetv3_small_100", num_classes = 2, dropout = 0.3):
    model = timm.create_model(model_name, pretrained = True)
    try:
        model.reset_classifier(num_classes=num_classes, dropout=dropout)
    except TypeError:
        model.reset_classifier(num_classes=num_classes)
    return model
