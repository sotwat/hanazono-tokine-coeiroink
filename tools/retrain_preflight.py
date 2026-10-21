"""Fail before training if the pilot checkpoint cannot be restored completely."""
from pathlib import Path
import json
import sys

import espnet
import numpy as np
import torch
import yaml
from espnet2.tasks.gan_tts import GANTTSTask


def main():
    run = Path(sys.argv[1])
    assert espnet.__version__ == '202211', espnet.__version__
    assert torch.__version__ == '1.12.1+cu113', torch.__version__
    assert np.__version__ == '1.23.5', np.__version__
    assert torch.cuda.is_available()
    args = GANTTSTask.get_parser().parse_args(['--config', str(run / 'train.yaml')])
    assert len(args.token_list) == 47
    assert args.token_list == yaml.safe_load((run / 'config180.yaml').read_text())['token_list']
    assert args.max_epoch == 185 and args.resume
    assert Path(args.output_dir) == run / 'model'
    preprocess = GANTTSTask.build_preprocess_fn(args, train=True)
    data = dict(line.split(maxsplit=1) for line in Path(args.train_data_path_and_name_and_type[0][0]).read_text().splitlines())
    for line in Path(args.train_shape_file[0]).read_text().splitlines():
        key, shape = line.split()
        tokens = preprocess(key, {'text': data[key]})['text']
        assert len(tokens) == int(shape.split(',')[0]), (key, len(tokens), shape)
    model = GANTTSTask.build_model(args)
    optimizers = GANTTSTask.build_optimizers(args, model)
    state = torch.load(run / 'model/checkpoint.pth', map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    epoch = state['reporter']['epoch']
    assert 180 <= epoch < 185, epoch
    assert len(optimizers) == len(state['optimizers']) == 2
    counts = []
    for optimizer, saved in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(saved)
        count = 0
        for group in optimizer.param_groups:
            for param in group['params']:
                moment = optimizer.state[param]
                assert moment['exp_avg'].shape == param.shape
                assert moment['exp_avg_sq'].shape == param.shape
                assert torch.isfinite(moment['exp_avg']).all()
                assert torch.isfinite(moment['exp_avg_sq']).all()
                count += 1
        counts.append(count)
    assert counts == [942, 106], counts
    report = {'epoch': epoch, 'espnet': espnet.__version__, 'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(), 'strict_model_load': True, 'optimizer_moment_shapes_valid': counts, 'phoneme_lengths_match': len(data), 'max_epoch': args.max_epoch, 'token_list': args.token_list, 'learning_rates': [[g['lr'] for g in o.param_groups] for o in optimizers]}
    (run / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
