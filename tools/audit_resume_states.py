"""Inspect saved optimizer continuity and preflight records without running training."""
import argparse
import json
from pathlib import Path
import re

import torch

p = argparse.ArgumentParser()
p.add_argument('recovered', type=Path)
p.add_argument('completed', type=Path)
p.add_argument('log', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
states = [torch.load(x, map_location='cpu', weights_only=False) for x in (a.recovered, a.completed)]
report = {'epochs': [s['reporter']['epoch'] for s in states], 'optimizers': []}
assert report['epochs'] == [180, 200]
for old, new in zip(states[0]['optimizers'], states[1]['optimizers']):
    assert old['param_groups'][0]['params'] == new['param_groups'][0]['params']
    steps = []
    for key, v in old['state'].items():
        n = new['state'][key]
        assert v['exp_avg'].shape == n['exp_avg'].shape
        assert v['exp_avg_sq'].shape == n['exp_avg_sq'].shape
        assert torch.isfinite(n['exp_avg']).all() and torch.isfinite(n['exp_avg_sq']).all()
        steps.append([float(v['step']), float(n['step'])])
    report['optimizers'].append({'state_count': len(steps), 'step_pairs': sorted(set(map(tuple, steps))),
        'step_deltas': sorted({y-x for x,y in steps}),
        'initial_lr': old['param_groups'][0]['lr'], 'final_lr': new['param_groups'][0]['lr']})
report['schedulers'] = [[{'last_epoch': x['last_epoch'], 'gamma': x['gamma']} for x in s['schedulers']] for s in states]
log = a.log.read_text()
report['preflights'] = [json.loads(line) for line in log.splitlines() if line.startswith('{"epoch":')]
report['resume_log_lines'] = [line for line in log.splitlines() if 'The training was resumed' in line]
report['traceback_count'] = log.count('Traceback (most recent call last)')
report['optimizer_failure_lines'] = [line for line in log.splitlines() if re.search(r'optimizer.*(fail|mismatch|fresh|error)', line, re.I)]
a.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
