from pathlib import Path
import argparse,json,shutil,marshal,subprocess,sys
from PyInstaller.archive.readers import CArchiveReader
from PyInstaller.archive.writers import CArchiveWriter
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--models',type=Path,required=True);parser.add_argument('--port',type=int,default=50033);args=parser.parse_args()
root=args.root.resolve(); runtime=root/'runtime'; eng=runtime/'engine';eng.mkdir(parents=True,exist_ok=True)
original=Path('/Applications/COEIROINKv2.app/Contents/MacOS/engine')
internal=eng/'_internal'
if not internal.exists():internal.symlink_to(original/'_internal',target_is_directory=True)
(eng/'engine_settings.json').write_text('{"device":"cpu"}')
source=Path('/Applications/COEIROINKv2.app/Contents/MacOS/speaker_info/hanazono-tokine-2.1.0');s=runtime/'speaker_info'/'audit';s.mkdir(parents=True,exist_ok=True)
for n in ['portrait.png','policy.md','LICENSE.txt']:
 if not (s/n).exists():(s/n).symlink_to(source/n)
models=args.models.resolve();epochs=sorted(int(x.stem[:-5]) for x in models.glob('*epoch.pth'))
styles=[]
for ep in epochs:
 sid=500000+ep;styles.append({'styleId':sid,'styleName':f'{ep}epoch'})
 m=s/'model'/str(sid);m.mkdir(parents=True,exist_ok=True)
 if not (m/f'{ep}epoch.pth').exists():(m/f'{ep}epoch.pth').symlink_to(models/f'{ep}epoch.pth')
 shutil.copy2(models/('config_181-200.yaml' if ep>=181 else 'config.yaml'),m/'config.yaml')
 for d,name,src in [('icons',f'{sid}.png',source/'icons/1100.png'),('portraits',f'{sid}.png',source/'portrait.png')]+[('voice_samples',f'{sid}_{i:03}.wav',source/f'voice_samples/1100_{i:03}.wav') for i in (1,2,3)]:
  (s/d).mkdir(exist_ok=True);dest=s/d/name
  if not dest.exists():dest.symlink_to(src)
uuid='8cb3363b-0292-4d86-bdf5-52949a66ee01'
(s/'metas.json').write_text(json.dumps({'speakerName':'Tokine audit','speakerUuid':uuid,'styles':styles,'version':'1.0.0'}))
archive=CArchiveReader(str(original/'engine'));chunks={n:archive.extract(n) for n in archive.toc}
code=marshal.loads(chunks['engine'])
# Change only the private audit copy's port. Installed application remains untouched.
code=code.replace(co_consts=tuple(args.port if x==50032 else x for x in code.co_consts));chunks['engine']=marshal.dumps(code)
class Writer(CArchiveWriter):
 def _write_entry(self,fp,e):
  n,_,c,t=e;return self._write_blob(fp,chunks.get(n,b''),n,t,compress=c)
entries=[(n,'',v[3],v[4]) for n,v in archive.toc.items()]+[(x,'',False,'o') for x in archive.options]
pkg=root/'audit.pkg';Writer(str(pkg),entries,'Python.framework/Versions/3.10/Python')
import PyInstaller
from PyInstaller.utils.osx import remove_signature_from_binary, fix_exe_for_code_signing
binary=eng/'engine';shutil.copy2(Path(PyInstaller.__file__).parent/'bootloader/Darwin-64bit/run',binary);remove_signature_from_binary(str(binary))
with binary.open('ab') as f:f.write(pkg.read_bytes())
fix_exe_for_code_signing(str(binary));binary.chmod(0o755)
subprocess.run(['codesign','--force','--sign','-',str(binary)],check=True,capture_output=True)
(root/'inventory.json').write_text(json.dumps({'epochs':epochs,'missing':[x for x in range(1,201) if x not in epochs],'uuid':uuid,'port':args.port},indent=2))
print('Prepared',len(epochs),'original models',binary)
