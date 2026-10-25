"""Prepare or run paired continuations from the untouched full epoch-180 state."""
import argparse
import copy
import hashlib
import io
import inspect
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
import shutil
import wave
import zipfile

import torch
import yaml

CHECKPOINT_SHA256 = 'ab6925215af0ea111939313749c64bab66743eef90411d9f30165c901687b097'
ARMS = {'control': 1.0, 'quarter_lr': 0.25}
SPLIT_ARM = 'discriminator_quarter'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    kwargs = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location='cpu', **kwargs)


def compare_states(original, candidate, factor):
    factors = [factor, factor] if isinstance(factor, (int, float)) else list(factor)
    assert len(factors) == 2
    assert len(original['optimizers']) == len(candidate['optimizers']) == 2
    assert len(original['schedulers']) == len(candidate['schedulers']) == 2
    assert candidate['reporter']['epoch'] == original['reporter']['epoch'] == 180
    assert len(original['model']) == len(candidate['model']) == 1049
    assert all(torch.equal(v, candidate['model'][k]) for k, v in original['model'].items())
    counts = []
    for a, b, scale in zip(original['optimizers'], candidate['optimizers'], factors):
        assert a['state'].keys() == b['state'].keys()
        counts.append(len(a['state']))
        for pid, state in a['state'].items():
            for key, value in state.items():
                actual = b['state'][pid][key]
                assert torch.equal(value, actual) if torch.is_tensor(value) else value == actual
                if torch.is_tensor(actual):
                    assert torch.isfinite(actual).all()
        assert len(a['param_groups']) == len(b['param_groups'])
        for ga, gb in zip(a['param_groups'], b['param_groups']):
            assert ga.keys() == gb.keys()
            for key, value in ga.items():
                assert gb[key] == (value * scale if key in ('lr', 'initial_lr') else value)
    assert counts == [942, 106]
    for a, b, scale in zip(original['schedulers'], candidate['schedulers'], factors):
        assert a.keys() == b.keys()
        for key, value in a.items():
            assert b[key] == ([x * scale for x in value] if key in ('base_lrs', '_last_lr') else value)
    return {'model_tensors_identical': 1049, 'optimizer_states_identical': counts,
            'epoch': 180, 'lr_factor': factor,
            'learning_rates': [o['param_groups'][0]['lr'] for o in candidate['optimizers']]}


def prepare_split(args):
    """Reuse verified pilot data; change only the discriminator LR trajectory."""
    pilot, reference, destination = (p.resolve() for p in
                                   (args.pilot, args.reference, args.destination))
    assert not destination.exists(), 'Refusing to overwrite an existing experiment'
    source = reference / 'checkpoint180_recovered.pth'
    assert digest(source) == CHECKPOINT_SHA256
    original = load(source)
    preparation = json.loads((pilot / 'preparation.json').read_text())
    assert preparation['checkpoint_sha256'] == CHECKPOINT_SHA256
    assert len(preparation['audio']) == 652
    config = yaml.safe_load((pilot / 'control/train.yaml').read_text())
    original_config = copy.deepcopy(config)
    assert config['max_epoch'] == 182 and config['optim_conf']['lr'] == .0001
    assert config['optim2_conf']['lr'] == .0001 and config['generator_first'] is False
    assert len(config['token_list']) == 47
    for item in preparation['audio'].values():
        assert digest(Path(item['path'])) == item['sha256']
    candidate = copy.deepcopy(original)
    for group in candidate['optimizers'][1]['param_groups']:
        for key in ('lr', 'initial_lr'):
            if key in group:
                group[key] *= .25
    for key in ('base_lrs', '_last_lr'):
        if key in candidate['schedulers'][1]:
            candidate['schedulers'][1][key] = [v * .25 for v in candidate['schedulers'][1][key]]
    report = compare_states(original, candidate, [1., .25])
    folder = destination / SPLIT_ARM
    modeldir = folder / 'model'
    config['output_dir'] = str(modeldir)
    config['optim2_conf']['lr'] *= .25
    differences = [k for k in config if config[k] != original_config[k]]
    assert set(differences) == {'output_dir', 'optim2_conf'}
    modeldir.mkdir(parents=True)
    torch.save(candidate, modeldir / 'checkpoint.pth')
    compare_states(original, load(modeldir / 'checkpoint.pth'), [1., .25])
    torch.save(original['model'], modeldir / '180epoch.pth')
    shutil.copy2(reference / 'config180.yaml', folder / 'config180.yaml')
    (folder / 'train.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    preparation.update(status='prepared_not_started', arms={SPLIT_ARM: report},
                       source_pilot=str(pilot), config_difference_keys=differences,
                       intervention='Generator LR unchanged; discriminator LR and scheduler base divided by four.')
    (destination / 'preparation.json').write_text(json.dumps(preparation, ensure_ascii=False, indent=2))
    print(json.dumps(report), flush=True)


def prepare(args):
    reference = args.reference.resolve()
    destination = args.destination.resolve()
    assert not destination.exists(), 'Refusing to overwrite an existing experiment'
    checkpoint = reference / 'checkpoint180_recovered.pth'
    assert digest(checkpoint) == CHECKPOINT_SHA256
    original = load(checkpoint)
    config = yaml.safe_load((reference / 'config180.yaml').read_text())
    assert config['version'] == '202211' and len(config['token_list']) == 47
    assert config['num_iters_per_epoch'] == 1000
    speech_shapes = dict(l.split() for l in (reference / 'train_speech_shape180').read_text().splitlines())
    text_shapes = dict(l.split() for l in (reference / 'train_text_shape180').read_text().splitlines())
    corpus = dict(l.split(':', 1) for l in args.corpus.read_text().splitlines() if l.strip())
    assert len(speech_shapes) == len(text_shapes) == 652
    destination.mkdir(parents=True)
    wavdir = destination / 'data/wavs'
    wavdir.mkdir(parents=True)
    audio = {}
    with zipfile.ZipFile(args.audio_zip) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith('.wav') and not n.startswith('__MACOSX/')]
        assert len(members) == 652 and len({Path(n).stem for n in members}) == 652
        for name in members:
            key = Path(name).stem
            assert key in speech_shapes and key in corpus
            content = archive.read(name)
            with wave.open(io.BytesIO(content)) as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (44100, 1, 2)
                assert wav.getnframes() == int(speech_shapes[key].split(',')[0])
            target = wavdir / (key + '.wav')
            target.write_bytes(content)
            audio[key] = {'path': str(target), 'sha256': hashlib.sha256(content).hexdigest()}
    ids = sorted(audio)
    for split, keys in [('train', ids), ('valid', ids[:5])]:
        folder = destination / 'data' / split
        folder.mkdir()
        for name, values in [('speech_shape', speech_shapes), ('text_shape.phn', text_shapes), ('text', corpus)]:
            (folder / name).write_text(''.join(f'{k} {values[k]}\n' for k in keys))
        (folder / 'wav.scp').write_text(''.join(f'{k} {audio[k]["path"]}\n' for k in keys))
        config[split + '_data_path_and_name_and_type'] = [[str(folder / 'text'), 'text', 'text'], [str(folder / 'wav.scp'), 'speech', 'sound']]
        config[split + '_shape_file'] = [str(folder / 'text_shape.phn'), str(folder / 'speech_shape')]
    for key in ('config', 'version', 'required', 'distributed'):
        config.pop(key, None)
    # All pilot checkpoints stay available; quality selection happens after synthesis.
    config.update(max_epoch=182, resume=True, init_param=[], keep_nbest_models=1000)
    reports = {}
    for arm, factor in ARMS.items():
        folder = destination / arm
        modeldir = folder / 'model'
        modeldir.mkdir(parents=True)
        candidate = copy.deepcopy(original)
        for optimizer in candidate['optimizers']:
            for group in optimizer['param_groups']:
                for key in ('lr', 'initial_lr'):
                    if key in group:
                        group[key] *= factor
        for scheduler in candidate['schedulers']:
            for key in ('base_lrs', '_last_lr'):
                if key in scheduler:
                    scheduler[key] = [v * factor for v in scheduler[key]]
        torch.save(candidate, modeldir / 'checkpoint.pth')
        reports[arm] = compare_states(original, load(modeldir / 'checkpoint.pth'), factor)
        torch.save(original['model'], modeldir / '180epoch.pth')
        arm_config = copy.deepcopy(config)
        arm_config['output_dir'] = str(modeldir)
        # The checkpoint overrides optimizer/scheduler values on resume.
        for key in ('optim_conf', 'optim2_conf'):
            arm_config[key]['lr'] *= factor
        (folder / 'train.yaml').write_text(yaml.safe_dump(arm_config, allow_unicode=True, sort_keys=False))
        shutil.copy2(reference / 'config180.yaml', folder / 'config180.yaml')
    controls = yaml.safe_load((destination / 'control/train.yaml').read_text())
    treatment = yaml.safe_load((destination / 'quarter_lr/train.yaml').read_text())
    differences = [k for k in controls if controls[k] != treatment[k]]
    assert set(differences) == {'output_dir', 'optim_conf', 'optim2_conf'}, differences
    report = {'status': 'prepared_not_started', 'checkpoint_sha256': CHECKPOINT_SHA256,
              'preparation_environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': importlib.metadata.version('numpy')},
              'audio_zip_sha256': digest(args.audio_zip), 'audio': audio, 'arms': reports,
              'config_difference_keys': differences, 'max_epoch': 182,
              'validation_overlap': 'All five validation utterances were in the original training set; not held-out evidence.',
              'quality_gate': 'Unprocessed speech versus original 180 and untreated 200; no automatic extension or installation.'}
    (destination / 'preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'audio'}, indent=2))


def stage_runtime_audio(run_dir, config, preparation, cache_root=Path('/content/tokine-controlled-cache')):
    """Read training WAVs from verified local bytes instead of Drive FUSE workers."""
    cache = cache_root / preparation['audio_zip_sha256'][:16]
    wavdir = cache / 'wavs'
    wavdir.mkdir(parents=True, exist_ok=True)
    expected = preparation['audio']
    missing = [key for key, item in expected.items()
               if not (wavdir / (key + '.wav')).is_file()
               or digest(wavdir / (key + '.wav')) != item['sha256']]
    if missing:
        archives = list(run_dir.parent.parent.parent.glob('*/花園トキネtalk_full_wavs.zip'))
        assert len(archives) == 1, archives
        archive_path = cache / 'source.zip'
        shutil.copyfile(archives[0], archive_path)
        assert digest(archive_path) == preparation['audio_zip_sha256']
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith('.wav') or name.startswith('__MACOSX/'):
                    continue
                key = Path(name).stem
                assert key in expected
                content = archive.read(name)
                assert hashlib.sha256(content).hexdigest() == expected[key]['sha256']
                (wavdir / (key + '.wav')).write_bytes(content)
    runtime = copy.deepcopy(config)
    for key, item in expected.items():
        assert digest(wavdir / (key + '.wav')) == item['sha256'], key
    for split in ('train', 'valid'):
        entries = runtime[split + '_data_path_and_name_and_type']
        speech = next(entry for entry in entries if entry[1] == 'speech')
        ids = [line.split()[0] for line in Path(speech[0]).read_text().splitlines()]
        scp = cache / (split + '.scp')
        scp.write_text(''.join(f'{key} {wavdir / (key + ".wav")}\n' for key in ids))
        speech[0] = str(scp)
    target = cache / (run_dir.name + '.yaml')
    target.write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False))
    print('LOCAL_AUDIO_VERIFIED', len(expected), flush=True)
    return target


def run(args):
    import espnet
    import numpy as np
    from espnet2.tasks.gan_tts import GANTTSTask
    import espnet2.train.trainer as trainer

    assert espnet.__version__ == '202211'
    assert torch.__version__ == '1.12.1+cu113'
    assert np.__version__ == '1.23.5'
    assert torch.cuda.is_available(), 'No CUDA GPU: training has not started'
    torch.set_num_threads(1)
    run_dir = args.run_dir.resolve()
    preparation = json.loads((run_dir.parent / 'preparation.json').read_text())
    assert run_dir.name in (*ARMS, SPLIT_ARM)
    config_path = run_dir / 'train.yaml'
    config = yaml.safe_load(config_path.read_text())
    assert config['max_epoch'] == 182 and config['num_iters_per_epoch'] == 1000
    target_epoch = args.max_epoch
    assert 182 <= target_epoch <= 200
    assert Path(config['output_dir']) == run_dir / 'model'
    assert config['resume'] and not config['init_param'] and not config['use_amp']
    state = load(run_dir / 'model/checkpoint.pth')
    epoch = state['reporter']['epoch']
    assert len(state['model']) == 1049 and all(torch.isfinite(v).all() for v in state['model'].values())
    assert [len(o['state']) for o in state['optimizers']] == [942, 106]
    assert len(state['schedulers']) == 2
    assert 180 <= epoch < target_epoch, f'Run is already complete or invalid: {epoch}'
    expected_lr = preparation['arms'][run_dir.name]['learning_rates']
    for i, optimizer in enumerate(state['optimizers']):
        assert abs(optimizer['param_groups'][0]['lr'] - expected_lr[i] * .999875 ** (epoch - 180)) < 1e-14
        for moment in optimizer['state'].values():
            step = moment['step']
            assert float(step) == epoch * 1000
    assert all(s['last_epoch'] == epoch for s in state['schedulers'])
    runtime_config = stage_runtime_audio(run_dir, config, preparation)
    runtime = yaml.safe_load(runtime_config.read_text())
    runtime['max_epoch'] = target_epoch
    runtime_config.write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False))
    parsed = GANTTSTask.get_parser().parse_args(['--config', str(runtime_config)])
    model = GANTTSTask.build_model(parsed)
    model.load_state_dict(state['model'], strict=True)
    optimizers = GANTTSTask.build_optimizers(parsed, model)
    for optimizer, saved in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(saved)
        for group in optimizer.param_groups:
            for p in group['params']:
                for key in ('exp_avg', 'exp_avg_sq'):
                    assert optimizer.state[p][key].shape == p.shape
                    assert torch.isfinite(optimizer.state[p][key]).all()
    preprocess = GANTTSTask.build_preprocess_fn(parsed, train=True)
    text = dict(l.split(maxsplit=1) for l in Path(parsed.train_data_path_and_name_and_type[0][0]).read_text().splitlines())
    for line in Path(parsed.train_shape_file[0]).read_text().splitlines():
        key, shape = line.split()
        assert len(preprocess(key, {'text': text[key]})['text']) == int(shape.split(',')[0]), key
    assert parsed.token_list == yaml.safe_load((run_dir / 'config180.yaml').read_text())['token_list']
    preflight = {'epoch': epoch, 'target_epoch': target_epoch, 'gpu': torch.cuda.get_device_name(), 'arm': run_dir.name,
                 'token_count': 47, 'phoneme_lengths_verified': len(text),
                 'learning_rates': [o.param_groups[0]['lr'] for o in optimizers],
                 'source_checkpoint_sha256': preparation['checkpoint_sha256']}
    (run_dir / 'preflight.json').write_text(json.dumps(preflight, indent=2))
    (run_dir / f'preflight_{epoch}_to_{target_epoch}.json').write_text(json.dumps(preflight, indent=2))
    print('CONTROLLED_PREFLIGHT_OK', json.dumps(preflight), flush=True)
    del model, optimizers, state
    import gc
    gc.collect()

    # Reporter history references unavailable pre-180 weights. Do not average or
    # call the latest training count a quality winner at the end of the pilot.
    def skip_average(*a, **kw):
        print('PILOT_AVERAGING_DISABLED: retain individual unmodified epochs', flush=True)
    trainer.average_nbest_models = skip_average
    # Preflight imports can install a WARNING-level root handler before ESPnet's
    # basicConfig call. Restore the configured level so batch progress is visible.
    logging.getLogger().setLevel(getattr(logging, parsed.log_level))
    GANTTSTask.main(cmd=['--config', str(runtime_config)])
    completed = load(run_dir / 'model/checkpoint.pth')
    assert completed['reporter']['epoch'] == target_epoch
    assert all((run_dir / f'model/{e}epoch.pth').is_file() for e in range(180, target_epoch + 1))
    assert all(torch.isfinite(v).all() for v in completed['model'].values())
    assert all(float(m['step']) == target_epoch * 1000 for o in completed['optimizers'] for m in o['state'].values())
    (run_dir / 'training_complete.json').write_text(json.dumps({'epoch': target_epoch, 'quality': 'not_yet_evaluated', 'arm': run_dir.name}, indent=2))
    print('PILOT_TRAINING_COMPLETE_QUALITY_NOT_YET_EVALUATED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    for name in ('reference', 'destination', 'audio-zip', 'corpus'):
        p.add_argument('--' + name, type=Path, required=True)
    p = commands.add_parser('prepare-split')
    for name in ('pilot', 'reference', 'destination'):
        p.add_argument('--' + name, type=Path, required=True)
    p = commands.add_parser('run')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--max-epoch', type=int, choices=range(182, 201), default=182)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'prepare-split':
        prepare_split(args)
    else:
        run(args)
