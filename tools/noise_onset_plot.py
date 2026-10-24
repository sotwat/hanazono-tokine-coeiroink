"""Plot periodic artifact evidence; no broadband HF energy curve is used."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    summary = json.loads((root / 'onset_summary.json').read_text())
    rows = json.loads((root / 'results.json').read_text())
    epochs = summary['epochs']
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout='constrained')
    colors = {'user': '#24599b', 'travel': '#a55426'}
    for test, color in colors.items():
        selected = sorted((x for x in rows if x['test'] == test), key=lambda x: x['epoch'])
        axes[0].scatter([x['epoch'] for x in selected], [x['comb_mean_db'] for x in selected],
                        label=test, s=12, color=color)
    axes[0].set_ylabel('Periodic local prominence (dB)')
    axes[0].legend()
    axes[1].bar([x['epoch'] for x in epochs], [x['shared_lines_over_3db'] for x in epochs],
                color='#24599b', width=.85)
    axes[1].set_ylabel('Shared lines > 3 dB\n(two ordinary texts)')
    values = np.full((17, 200), np.nan)
    for epoch in epochs:
        selected = [x for x in rows if x['epoch'] == epoch['epoch']]
        values[:, epoch['epoch'] - 1] = np.mean([x['comb_line_scores_db'] for x in selected], axis=0)
    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad('#e5e5e5')
    im = axes[2].imshow(values, origin='lower', aspect='auto', interpolation='nearest',
                        extent=(.5, 200.5, 9.5 * 44100 / 64000, 26.5 * 44100 / 64000),
                        norm=matplotlib.colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=10), cmap=cmap)
    axes[2].set_ylabel('Fixed grid frequency (kHz)')
    axes[2].set_xlabel('Saved original model epoch (grey columns: unavailable)')
    fig.colorbar(im, ax=axes[2], label='Mean local prominence (dB)')
    for ax in axes:
        ax.axvline(180.5, color='#333333', linestyle=':', linewidth=1)
        ax.set_xlim(.5, 200.5)
    axes[0].grid(alpha=.2)
    axes[1].grid(axis='y', alpha=.2)
    fig.suptitle('Tokine: 150 original models, two matched ordinary texts')
    svg = root / 'periodic_noise_onset.svg'
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    fig.savefig(root / 'periodic_noise_onset.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    plot(parser.parse_args().root)
