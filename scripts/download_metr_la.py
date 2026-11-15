"""Download METR-LA from the public Zenodo traffic dataset record."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

URL='https://zenodo.org/records/5146275/files/METR-LA.csv?download=1'
EXPECTED_MD5='d3adaa79856bf610f25558fc242c7190'
DEST=Path(__file__).resolve().parents[1]/'data'/'raw'/'METR-LA.csv'

def main():
 DEST.parent.mkdir(parents=True,exist_ok=True)
 tmp=DEST.with_suffix('.csv.part')
 try:
  with urlopen(URL,timeout=30) as response,tmp.open('wb') as out:
   while True:
    chunk=response.read(1024*1024)
    if not chunk: break
    out.write(chunk)
  digest=hashlib.md5(tmp.read_bytes()).hexdigest()
  if digest!=EXPECTED_MD5: raise ValueError(f'MD5 mismatch: expected {EXPECTED_MD5}, got {digest}')
  tmp.replace(DEST); print(f'Downloaded and verified {DEST} ({DEST.stat().st_size} bytes).')
 except Exception:
  if tmp.exists(): tmp.unlink()
  raise
if __name__=='__main__': main()
