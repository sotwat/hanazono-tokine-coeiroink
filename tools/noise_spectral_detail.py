from pathlib import Path
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft,find_peaks
from scipy.ndimage import median_filter

def detail(wav,duration,min_prominence=4,peak_limit=8):
 fs,a=wavfile.read(wav);a=a.astype(float)/32768
 f,t,z=stft(a,fs,nperseg=2048,noverlap=1536,boundary=None,padded=False);p=np.abs(z)**2
 voiced=np.zeros(len(t),bool)
 for m in duration:
  for ph in m['phonemePitches']:
   r=ph['wavRange']
   if ph['phoneme'] in ['a','i','u','e','o']:
    voiced|=(t>r['start']/fs+.025)&(t<r['end']/fs-.025)
 if voiced.sum()<3:return {'peaks':[]}
 norm=p[:,voiced]/(p[(f>=300)&(f<=4000)][:,voiced].sum(axis=0)+1e-30)
 avg=10*np.log10(norm.mean(axis=1)+1e-30);med=10*np.log10(np.median(norm,axis=1)+1e-30)
 smooth=median_filter(med,size=21);prom=med-smooth;inds,_=find_peaks(prom,height=min_prominence,distance=5)
 inds=[i for i in inds if 6000<=f[i]<=18000 and med[i]>-65]
 peaks=[]
 for i in sorted(inds,key=lambda i:prom[i],reverse=True)[:peak_limit]:
  local=10*np.log10(norm[i]+1e-30)-np.median(10*np.log10(norm[max(i-10,0):i+11]+1e-30),axis=0)
  peaks.append({'hz':round(float(f[i]),1),'prominence_db':round(float(prom[i]),2),'median_relative_bin_db':round(float(med[i]),2),'frame_presence_over_6db':round(float(np.mean(local>=6)),3)})
  if peak_limit is None:
   peaks[-1].update(prominence_db_exact=float(prom[i]),frame_presence_over_6db_exact=float(np.mean(local>=6)))
 indices=[int(np.argmin(abs(f-hz))) for hz in np.arange(10,27)*fs/64]
 line_scores=[float(med[i]-np.median(med[np.r_[i-7:i-2,i+3:i+8]])) for i in indices]
 return {'peaks':peaks,'comb_mean_db':float(np.mean(line_scores)), 'comb_lines_over_3db':int(sum(x>3 for x in line_scores)), 'comb_line_scores_db':line_scores,'frequencies':f.tolist(),'mean_db':avg.tolist(),'median_db':med.tolist()}
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent);args=parser.parse_args();root=args.root;out={}
 for kind in ['original','installed']:
  for wav in sorted((root/kind).glob('*.wav')):
   durations=json.loads(wav.with_suffix('.json').read_text())['moraDurations'];out[f'{kind}/{wav.stem}']=detail(wav,durations)
 (root/'spectral_details.json').write_text(json.dumps(out,indent=2))
 for ep in [100,180,190,200]:
  for kind in ['original','installed']:
   for text in ['vowels','user']:
    k=f'{kind}/{ep:03}_{text}'
    if k in out:print(k,out[k]['peaks'][:4])
