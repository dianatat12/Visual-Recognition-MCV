import torch
import torch.nn as nn
import torchvision.models as models
import timm


def get_feature_extractor_model(model_name="resnet50", pretrained=True):
    model_functions = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "alexnet": models.alexnet,
        "vgg16": models.vgg16,
        "squeezenet1_0": models.squeezenet1_0,
        "densenet161": models.densenet161,
        "inception_v3": models.inception_v3,
        "googlenet": models.googlenet,
        "mobilenet_v2": models.mobilenet_v2,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "mobilenet_v3_small": models.mobilenet_v3_small,
        # transformer models
        "vit_base_patch16_224": timm.create_model,
        "vit_large_patch16_224": timm.create_model,
    }

    if model_name.startswith("vit_"):
        raise ValueError("ViT models are not supported for feature extraction.")
    elif model_name not in model_functions:
        raise ValueError(f"Model {model_name} not supported.")
    else:
        model = model_functions[model_name](pretrained=pretrained)

        if "resnet" in model_name or "vgg" in model_name or "densenet" in model_name:
            # Remove the final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            # Add AdaptiveAvgPool2d to convert to a fixed size feature map
            model.add_module("AdaptiveAvgPool", nn.AdaptiveAvgPool2d((1, 1)))
        elif model_name == "alexnet" or model_name.startswith("squeezenet"):
            # Remove the final classification layer
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        elif "mobilenet" in model_name:
            # Remove the final classification layer
            model = nn.Sequential(*(list(model.children())[:-1]))
            if "v3" in model_name:
                model[-1] = nn.Sequential(*(list(model[-1].children())[:-3]))
                model[-1].add_module("AdaptiveAvgPool", nn.AdaptiveAvgPool2d((1, 1)))
        elif model_name == "inception_v3" or model_name == "googlenet":
            # Remove the final classification layer
            model.fc = nn.Identity()
            if model_name == "inception_v3":
                model.AuxLogits.fc = nn.Identity()
            # Add AdaptiveAvgPool2d to convert to a fixed size feature map
            model.add_module("AdaptiveAvgPool", nn.AdaptiveAvgPool2d((1, 1)))

    return model
