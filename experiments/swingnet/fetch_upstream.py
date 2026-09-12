"""Fetch pinned upstream files for local experimentation; verify content hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent

def main():
    for spec in json.loads((HERE/'upstream/sources.json').read_text(encoding='utf8')):
        target=HERE/'upstream'/spec['file']
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=spec['sha256']:
                raise RuntimeError(f'Refusing to overwrite modified file: {target.name}')
            continue
        with urllib.request.urlopen(spec['url'],timeout=60) as response: data=response.read()
        if hashlib.sha256(data).hexdigest()!=spec['sha256']:raise RuntimeError('Upstream checksum mismatch')
        target.write_bytes(data)
        print('Verified',target.name)

if __name__=='__main__':main()
