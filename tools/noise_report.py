from pathlib import Path
import json,csv,argparse,statistics,subprocess,re,wave,array
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent);args=parser.parse_args();root=args.root
inv=json.loads((root/'inventory.json').read_text());original=json.loads((root/'original_results.json').read_text());installed=json.loads((root/'installed_results.json').read_text());spec=json.loads((root/'spectral_details.json').read_text())
rows=[]
for row in original+installed:
 key=f'{row["kind"]}/{row["epoch"]:03}_{row["test"]}';r={k:v for k,v in row.items() if k not in ['hf_psd_db','hf_freq_hz']};detail=spec[key];r.update({k:detail[k] for k in ['comb_mean_db','comb_lines_over_3db']});rows.append(r)
assert len({r['epoch'] for r in rows if r['kind']=='original' and r['test']=='vowels'})==len(inv['epochs'])
assert len(installed)==60 and len({r['epoch'] for r in installed})==20
with (root/'all_epoch_noise_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'metrics.json').write_text(json.dumps(rows,indent=2))
base=sorted([r for r in rows if r['kind']=='original' and r['test']=='vowels'],key=lambda r:r['epoch']);x=[r['epoch'] for r in base]
fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
axs[0].plot(x,[r['comb_mean_db'] for r in base],'.-',lw=1,color='#24599b');axs[0].set_ylabel('Periodic HF peak score (dB)');axs[0].axhline(0,color='#555555',lw=.6)
axs[1].plot(x,[r['vowel_hf8_rel_db'] for r in base],'.-',lw=1,color='#287a52');axs[1].set_ylabel('8-20 kHz / 0.3-4 kHz (dB)');axs[1].set_xlabel('Epoch (saved original checkpoints only)')
for ax in axs:
 ax.grid(alpha=.2);ax.set_xlim(0,202);ax.axvspan(181,200,color='#cc4736',alpha=.09)
fig.suptitle('Tokine: matched-text high-frequency screening (150 checkpoints)')
fig.savefig(root/'epoch_noise_overview.svg');fig.savefig(root/'epoch_noise_overview.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ep,color in [(100,'#24599b'),(180,'#287a52'),(190,'#dd8e25'),(200,'#bc3434')]:
 d=spec[f'installed/{ep:03}_user'];f=np.array(d['frequencies']);m=(f>=6000)&(f<=18000);axs[0].plot(f[m]/1000,np.array(d['median_db'])[m],label=str(ep),color=color,lw=.9)
for ep in [100,110,120,130,140,160,170,180,190,200]:
 selected=[r for r in rows if r['kind']=='installed' and r['epoch']==ep];scores=[r['comb_mean_db'] for r in selected];axs[1].scatter([ep]*len(scores),scores,s=20,color='#bc3434' if ep>=190 else '#24599b')
axs[0].set(xlabel='Frequency (kHz)',ylabel='Median vowel PSD / low-band power (dB)',title='Same user sentence, installed styles');axs[0].legend(title='Epoch',ncol=2);axs[0].grid(alpha=.2)
axs[1].set(xlabel='Epoch',ylabel='Periodic HF peak score (dB)',title='Installed styles: 3 texts each');axs[1].grid(alpha=.2)
fig.savefig(root/'hf_comparison.svg');fig.savefig(root/'hf_comparison.png',dpi=180);plt.close(fig)
# Equal-loudness listening comparison uses only linear gain, with the original WAVs kept intact.
parts=[];audio_stats=[]
for ep in [100,180,190,200]:
 src=root/'original'/f'{ep:03}_user.wav';dst=root/f'listen_{ep}.wav'
 r=subprocess.run(['ffmpeg','-hide_banner','-i',str(src),'-af','loudnorm=print_format=json','-f','null','-'],capture_output=True,text=True,check=True);l=json.loads(re.search(r'\{\s*"input_i"[\s\S]*?\}',r.stderr).group());gain=-23-float(l['input_i'])
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(src),'-af',f'volume={gain}dB','-c:a','pcm_s16le',str(dst)],check=True)
 with wave.open(str(dst),'rb') as w:assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,44100);b=w.readframes(w.getnframes());assert max(abs(x) for x in array.array('h',b))<32767;parts.append(b+b'\0'*88200)
 audio_stats.append({'epoch':ep,'original_lufs':l['input_i'],'linear_gain_db':gain,'target_lufs':-23})
with wave.open(str(root/'listen_100_180_190_200.wav'),'wb') as w:w.setparams((1,2,44100,0,'NONE','not compressed'));w.writeframes(b''.join(parts))
(root/'listening_levels.json').write_text(json.dumps(audio_stats,indent=2))
summary={'original_models':len(inv['epochs']),'installed_styles':20,'installed_texts':3,'rendered_wavs':len(original)+len(installed),'missing_epochs':inv['missing'],'pcm_clips':sum(r['pcm_clip_count'] for r in rows),'key_original_rows':[r for r in base if r['epoch'] in [100,170,175,178,179,180,181,182,183,184,185,190,200]],'installed_summaries':[]}
for ep in [100,110,120,130,140,160,170,180,190,200]:
 r=[x for x in rows if x['kind']=='installed' and x['epoch']==ep];summary['installed_summaries'].append({'epoch':ep,'mean_comb_db':statistics.mean(x['comb_mean_db'] for x in r),'min_comb_db':min(x['comb_mean_db'] for x in r),'max_comb_db':max(x['comb_mean_db'] for x in r),'mean_hf8_db':statistics.mean(x['vowel_hf8_rel_db'] for x in r)})
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2)[:13000])
