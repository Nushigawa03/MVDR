import torch
import torchaudio
import argparse

import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i','--audio', default='out/84-121123-0021_A.wav')
parser.add_argument('-o','--out', default='output.wav')
parser.add_argument('-m','--model', default='1900_236_16_200.ckpt')
args = parser.parse_args()

net = models.MVDRBeamformer().to(device)
net.load_state_dict(torch.load('models/'+args.model, map_location=device))

wa,sr=torchaudio.load(args.audio)
wa=net.forward(wa.to(device)).cpu().detach()[None]
torchaudio.save(filepath='out/'+args.out, src=wa, sample_rate=sr)
print("Output "+args.out)