"""Read Zenodo ZIP directory via HTTP ranges; extract only explicitly selected files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import time
import zipfile
import zlib
import requests

HERE = Path(__file__).resolve().parent
URL = 'https://zenodo.org/api/records/14677727/files/assets.zip/content'
SIZE = 14462954226


def get_range(start, end):
    for attempt in range(4):
        response = requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=(20,90), stream=True)
        if response.status_code == 206:
            data = response.content
            if len(data) != end-start+1:
                raise ValueError('Incomplete range')
            return data
        response.close()
        if response.status_code not in (429,500,502,503,504):
            raise RuntimeError(f'Range request returned {response.status_code}; refusing full archive download')
        time.sleep(2*(attempt+1))
    raise RuntimeError('Range retries exhausted')


def directory():
    tail_start = SIZE-131072
    tail = get_range(tail_start,SIZE-1)
    pos = tail.rfind(b'PK\x05\x06')
    if pos < 0: raise ValueError('Missing ZIP end')
    eocd = struct.unpack_from('<4s4H2IH',tail,pos)
    cdsize, cdoffset = eocd[5:7]
    if cdoffset == 0xffffffff or cdsize == 0xffffffff:
        loc = tail.rfind(b'PK\x06\x07',0,pos)
        zoff = struct.unpack_from('<IQI',tail,loc+4)[1]
        z = get_range(zoff,zoff+55)
        fields = struct.unpack('<4sQ2H2I4Q',z)
        cdsize,cdoffset = fields[-2:]
    cd = get_range(cdoffset,cdoffset+cdsize-1)
    # A standalone central-directory ZIP lets Python parse ZIP64 extra fields.
    end = struct.pack('<4s4H2IH',b'PK\x05\x06',0,0,0,0,len(cd),0,0)
    with zipfile.ZipFile(io.BytesIO(cd+end)) as archive:
        infos = archive.infolist()
    return [{'name':i.filename,'size':i.file_size,'compressed':i.compress_size,
             'offset':i.header_offset,'method':i.compress_type,'crc':i.CRC}
            for i in infos]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--extract',nargs='*')
    args=p.parse_args()
    listing=HERE/'upstream/zip-directory.json'
    listing.parent.mkdir(parents=True,exist_ok=True)
    if listing.exists():
        entries=json.loads(listing.read_text(encoding='utf8'))
    else:
        entries=directory()
        listing.write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf8')
    if not args.extract:
        for e in entries:
            if any(t in e['name'].lower() for t in ['.pth','.pt','.py','.ipynb','readme','license']):
                print(json.dumps(e,ensure_ascii=True))
        print(f'Total entries: {len(entries)}',flush=True)
        return
    for name in args.extract:
        entry=next(e for e in entries if e['name']==name)
        if entry['size']>500*1024**2: raise ValueError('Selected member exceeds 500 MiB')
        destination=HERE/'assets'/name
        if not destination.resolve().is_relative_to(HERE/'assets'): raise ValueError('Unsafe member path')
        if destination.exists():
            print(f'Exists {name}'); continue
        offset=entry['offset']
        header=get_range(offset,offset+29)
        if header[:4]!=b'PK\x03\x04': raise ValueError(f'Invalid local ZIP header at {offset}')
        fn,extra=struct.unpack_from('<HH',header,26)
        start=offset+30+fn+extra
        data=get_range(start,start+entry['compressed']-1)
        if entry['method']==8: data=zlib.decompress(data,-15)
        elif entry['method']!=0: raise ValueError('Unsupported ZIP compression')
        if len(data)!=entry['size'] or zlib.crc32(data)&0xffffffff!=entry['crc']: raise ValueError('ZIP CRC mismatch')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
        print(json.dumps({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}),flush=True)


if __name__=='__main__': main()
