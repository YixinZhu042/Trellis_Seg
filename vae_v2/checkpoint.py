"""Optional node-local staging for large optimizer checkpoints on shared storage."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil


def stage_checkpoint(source, cache_dir):
    source=Path(source).resolve()
    stat=source.stat()
    identity={'source':str(source),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache=Path(cache_dir).resolve();cache.mkdir(parents=True,exist_ok=True)
    target=cache/f'{key}.pt'
    with (cache/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            tmp=target.with_suffix('.tmp')
            with source.open('rb') as src,tmp.open('wb') as dst:
                shutil.copyfileobj(src,dst,64*1024*1024)
            after=source.stat()
            if after.st_size!=stat.st_size or after.st_mtime_ns!=stat.st_mtime_ns or tmp.stat().st_size!=stat.st_size:
                raise ValueError('Checkpoint changed while staging')
            tmp.replace(target)
        elif target.stat().st_size!=stat.st_size:
            raise ValueError('Incomplete local checkpoint cache')
    return target
