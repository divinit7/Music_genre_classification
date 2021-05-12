import gradio as gr
import torch
import numpy as np
from scipy import signal
import torchvision.models as models
from torchvision import transforms
import requests

import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import ToTensor
import sys
from PIL import Image
import logging

class DeviceDataLoader():
    def __init__(self, dl):
        self.dl = dl
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b)

    def __len__(self):
        return len(self.dl)

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained = True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)

    def forward(self, xb):
        return self.network(xb)
    
    

MODEL = Net()
MODEL.load_state_dict(torch.load("weights/model.pth"))
MODEL.eval()

labels = ('blues', 'classical', 'country', 'disco', 'hiphop' ,'jazz' ,'metal' ,'pop' ,'reggae' ,'rock')

def spectrogram(audio):
    sr, data = audio
    if len(data.shape) == 2:
        data = np.mean(data, axis=0)
    frequencies, times, spectrogram_data = signal.spectrogram(data, sr, window="hamming")
    plt.pcolormesh(times, frequencies, np.log10(spectrogram_data))
    return plt

def predict(input):
    preds = []
    try:
        image = spectrogram(input)
        
        transforms_image = transforms.Compose([
                T.Resize(image_size),
                        T.ToTensor()
                ])
        image = transforms_image(image)
        image = image.unsqueeze(0)
        
        output = Net(image)
        _,  preds = torch.max(output, dim =1)
    
    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]        
    print(preds)
    return preds

inputs = gr.inputs.Audio(source="upload")
outputs = gr.outputs.Label(num_top_classes=3)
interface = gr.Interface(fn=predict, inputs=inputs, outputs=outputs)

interface.launch()