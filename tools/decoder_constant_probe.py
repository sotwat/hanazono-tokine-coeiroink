"""Measure decoder-created periodic output with a zero latent input.

Functional forward follows the ESPnet HiFiGAN implementation (Apache-2.0,
Tomoki Hayashi). Diagnostic output is not a synthesized speech quality score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

def decode(state, z):
    prefix = 'tts.generator.decoder.'
    weights = {}
    for k in state:
        if k.startswith(prefix) and k.endswith('weight_v'):
            v = state[k]
            weights[k[:-8] + 'weight'] = v * state[k[:-8] + 'weight_g'] / torch.linalg.vector_norm(v, dim=(1,2), keepdim=True)
    def conv(x, name, dilation=1):
        k = weights[prefix + name + '.weight']
        return F.conv1d(x, k, state[prefix + name + '.bias'], padding=(k.shape[-1]-1)//2*dilation, dilation=dilation)
    c = conv(z, 'input_conv')
    for i, stride in enumerate([8,8,2,2,2]):
        c = F.conv_transpose1d(F.leaky_relu(c, .1), weights[prefix + f'upsamples.{i}.1.weight'], state[prefix + f'upsamples.{i}.1.bias'], stride=stride, padding=stride//2)
        blocks = []
        for j in range(3):
            x = c
            for idx, dilation in enumerate([1,3,5]):
                r = conv(F.leaky_relu(x, .1), f'blocks.{i*3+j}.convs1.{idx}.1', dilation)
                r = conv(F.leaky_relu(r, .1), f'blocks.{i*3+j}.convs2.{idx}.1')
                x = x + r
            blocks.append(x)
        c = sum(blocks)/3
    return torch.tanh(conv(F.leaky_relu(c, .01), 'output_conv.1'))

def main(root):
    torch.set_num_threads(4)
    out = []
    for item in json.loads((root / 'manifest.json').read_text()):
        state = torch.load(item['path'], map_location='cpu', weights_only=True)
        with torch.no_grad():
            v = decode(state, torch.zeros(1,192,64)).flatten().numpy()[8192:-8192]
        period = v.reshape(-1,512).mean(0)
        spectrum = np.fft.rfft(period)
        p = abs(spectrum)**2
        high = np.fft.rfftfreq(512, 1/44100) >= 6000
        out.append({'name': item['name'], 'zero_input_rms': float(np.sqrt(np.mean(v*v))),
                    'periodic_ac_rms': float(np.std(period)),
                    'high_periodic_power': float(p[high].sum()),
                    'peaks': [{'hz': float(i*44100/512), 'amplitude': float(2*abs(spectrum[i])/512)} for i in np.argsort(p[1:])[::-1][:8]+1]})
        (root / 'constant_probe.json').write_text(json.dumps(out, indent=2))
        print(item['name'], out[-1]['periodic_ac_rms'], out[-1]['high_periodic_power'], flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
