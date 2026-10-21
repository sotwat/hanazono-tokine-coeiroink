import argparse,base64,hashlib,json,time,urllib.request,wave,io
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

TEXTS={
 'vowels':'あー、いー、うー、えー、おー。今日はいい天気ですね。',
 'user':'迷惑かけちゃってごめんね、こんなに長く外にいるつもりはなかったんだよ',
 'travel':'東京から京都まで、新幹線で旅行します。切符を忘れないでください。',
}
def post(port,route,data):
 request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/{route}',json.dumps(data).encode(),{'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=240) as r:return r.read()
def analyze(path,durations):
 fs,a=wavfile.read(path)
 assert a.dtype==np.int16, 'Expected 16-bit PCM WAV'
 a=a.astype(float)/32768
 assert fs==44100 and a.ndim==1 and np.isfinite(a).all() and np.max(np.abs(a))>0
 n=2048;f,t,z=stft(a,fs,nperseg=n,noverlap=n-512,boundary=None,padded=False);power=np.abs(z)**2
 voice=np.zeros(t.shape,dtype=bool);pause=voice.copy()
 for m in durations:
  for ph in m['phonemePitches']:
   p=ph['phoneme'];r=ph['wavRange'];start=r['start']/fs;end=r['end']/fs
   if p in ('a','i','u','e','o'):
    voice|=(t>=start+.025)&(t<=end-.025)
   if p in ('pau','sil'):
    pause|=(t>=start+.06)&(t<=end-.06)
 if voice.sum()<3:voice=np.sum(power,axis=0)>np.max(np.sum(power,axis=0))*.2
 low=(f>=300)&(f<=4000);high=(f>=8000)&(f<=20000);higher=(f>=12000)&(f<=20000)
 def ratio(mask,band):
  if not mask.any():return None
  return float(10*np.log10((power[band][:,mask].sum()+1e-30)/(power[low][:,voice].sum()/voice.sum()*mask.sum()+1e-30)))
 spectra=power[high][:,voice];mean=spectra.mean(axis=1);flat=np.exp(np.mean(np.log(spectra+1e-30),axis=0))/(spectra.mean(axis=0)+1e-30)
 peak=int(np.argmax(mean));padded=np.maximum(mean,1e-30);tone=float(10*np.log10(padded[peak]/np.median(padded)))
 return {'seconds':len(a)/fs,'peak_dbfs':float(20*np.log10(np.max(np.abs(a)))),'rms_dbfs':float(20*np.log10(np.sqrt(np.mean(a*a)))),'pcm_clip_count':int(np.sum((a>=32767/32768)|(a<=-1))),'voiced_frames':int(voice.sum()),'vowel_hf8_rel_db':ratio(voice,high),'vowel_hf12_rel_db':ratio(voice,higher),'pause_hf8_rel_db':ratio(pause,high),'hf_flatness':float(np.median(flat)),'hf_peak_hz':float(f[high][peak]),'hf_peak_to_median_db':tone,'hf_psd_db':(10*np.log10(mean/(power[low][:,voice].sum()/voice.sum())+1e-30)).tolist(),'hf_freq_hz':f[high].tolist()}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent);parser.add_argument('--epochs',nargs='*',type=int);parser.add_argument('--texts',nargs='+',default=['vowels']);parser.add_argument('--port',type=int,default=50033);parser.add_argument('--installed',action='store_true');args=parser.parse_args();root=args.root
 inv=json.loads((root/'inventory.json').read_text());epochs=args.epochs or inv['epochs'];kind='installed' if args.installed else 'original';folder=root/kind;folder.mkdir(exist_ok=True);out=root/f'{kind}_results.json';results=json.loads(out.read_text()) if out.exists() else []
 for ep in epochs:
  for name in args.texts:
   old=next((x for x in results if x['epoch']==ep and x['test']==name),None)
   if old:continue
   started=time.time();stem=folder/f'{ep:03}_{name}';wav=stem.with_suffix('.wav');dur=stem.with_suffix('.json')
   if not wav.exists() or not dur.exists():
    req={'speakerUuid':('6cf172a4-b091-11f1-ace8-0242ac1c000c' if args.installed else inv['uuid']),'styleId':(1000 if args.installed else 500000)+ep,'text':TEXTS[name],'speedScale':1}
    response=json.loads(post(args.port,'predict_with_duration',req));wav.write_bytes(base64.b64decode(response.pop('wavBase64')));response['request']=req;dur.write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n')
   response=json.loads(dur.read_text());row={'epoch':ep,'test':name,'kind':kind,**analyze(wav,response['moraDurations'])};results.append(row);out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
   print(f'{kind} {ep:3} {name}: HF8={row["vowel_hf8_rel_db"]:.2f}dB duration={time.time()-started:.1f}s ({len(results)})',flush=True)
if __name__=='__main__':main()
