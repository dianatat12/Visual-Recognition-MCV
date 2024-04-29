from model import ResNet50, BERT_Module
import torch


resnet = ResNet50(finetune=False)
bert = BERT_Module(finetune=False)


def get_image_encoder_output_size(resnet):
    random_input = torch.randn(1, 3, 224, 224)
    resnet_output = resnet(random_input)

    return resnet_output.shape[0]


def get_text_encoder_output_size(bert):
    input_text = "This is a sample sentence."
    bert_output = bert(input_text)

    return bert_output.shape[0]


print(get_image_encoder_output_size(resnet))
print(get_text_encoder_output_size(bert))
