"""Build measured, equal-loudness comparisons for the selected inference repair."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.io import wavfile

from retrain_noise_compare import loudness

KINDS = ['180_original', '200_retrained', 'r200_balance0.5']

def build(root):
    rows = json.loads((root/'results.json').read_text()) + json.loads((root/'validation_results.json').read_text())
    names = list(dict.fromkeys(r['test'] for r in rows))
    listening = root/'listening'
    listening.mkdir(exist_ok=True)
    levels = []
    for name in names:
        parts = []
        for kind in KINDS:
            source = root/'raw'/kind/(name+'.wav')
            subprocess.run(['python3','/Users/aaa/.codex/skills/ffmpeg-skill/scripts/probe.py',str(source),'--compact'],capture_output=True,check=True)
            output = listening/(name+'_'+kind+'.wav')
            gain = -23-float(loudness(source)['input_i'])
            subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-af',f'volume={gain}dB','-c:a','pcm_s16le',str(output)],check=True)
            after = loudness(output)
            fs, pcm = wavfile.read(output)
            assert fs==44100 and pcm.dtype==np.int16 and pcm.ndim==1
            assert np.max(abs(pcm.astype(int)))<32767
            assert abs(float(after['input_i'])+23)<.12 and float(after['input_tp'])<-1
            levels.append({'test':name,'kind':kind,'gain_db':gain,'lufs':float(after['input_i']),'true_peak_dbtp':float(after['input_tp'])})
            parts.append(pcm)
        for suffix, indices in [('180_retrained_repaired',[0,1,2]),('retrained_before_after',[1,2])]:
            selected = []
            for i in indices:
                if selected:
                    selected.append(np.zeros(44100,dtype=np.int16))
                selected.append(parts[i])
            target = listening/(name+'_'+suffix+'.wav')
            wavfile.write(target,44100,np.concatenate(selected))
            subprocess.run(['python3','/Users/aaa/.codex/skills/ffmpeg-skill/scripts/probe.py',str(target),'--compact'],capture_output=True,check=True)
    (root/'listening_levels.json').write_text(json.dumps(levels,indent=2))
    fields = ['name','test','comb_mean_db','vowel_hf8_rel_db','rms_dbfs','pcm_clip_count','seconds']
    with (root/'interventions.csv').open('w') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    summary = {'texts':names,'model_count':13,'waveforms_measured':len(rows),'averages':{},'pairs':[],
               'listening_lufs_range':[min(x['lufs'] for x in levels),max(x['lufs'] for x in levels)],
               'listening_max_true_peak_dbtp':max(x['true_peak_dbtp'] for x in levels)}
    for kind in KINDS:
        selected=[r for r in rows if r['name']==kind]
        assert len(selected)==8
        summary['averages'][kind]=float(np.mean([r['comb_mean_db'] for r in selected]))
    for name in names:
        selected={r['name']:r for r in rows if r['test']==name}
        before,after=selected['200_retrained'],selected['r200_balance0.5']
        assert after['comb_mean_db']<before['comb_mean_db']
        summary['pairs'].append({'text':name,'e180':selected['180_original']['comb_mean_db'],
            'r200':before['comb_mean_db'],'fixed':after['comb_mean_db']})
    transcripts={}
    for filename in ['asr_baselines.jsonl','asr_baselines_more.jsonl','asr_candidates.jsonl','asr_validation.jsonl']:
        for line in (root/filename).read_text().splitlines():
            item=json.loads(line);path=Path(item['file']);transcripts[(path.parent.name,path.stem)]=item['transcript']
    summary['asr_comparison']=[]
    normalize=lambda s:re.sub(r'[^\w]','',s)
    for name in names:
        before=transcripts[('200_retrained',name)];after=transcripts[('r200_balance0.5',name)]
        assert normalize(before)==normalize(after),(name,before,after)
        summary['asr_comparison'].append({'text':name,'before':before,'after':after,'same_words_ignoring_punctuation':True})
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    build(p.parse_args().root)
