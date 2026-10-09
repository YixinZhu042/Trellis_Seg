"""Short GPU/DDP acceptance check for the VAE environment; no dataset changes."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist


def main():
    out = Path(sys.argv[1])
    rank, local, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    def mark(stage, **extra):
        record = dict(rank=rank, world=world, stage=stage, **extra)
        (out/f'rank{rank:02d}.json').write_text(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    try:
        mark('initialized')
        value = torch.tensor(float(rank+1), device=device)
        dist.all_reduce(value)
        assert value.item() == world*(world+1)/2
        mark('all_reduce_passed')
        values = [torch.empty(1, dtype=torch.int, device=device) for _ in range(world)]
        dist.all_gather(values, torch.tensor([rank], dtype=torch.int, device=device))
        assert [v.item() for v in values] == list(range(world))
        mark('all_gather_int_passed')
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Linear(32, 4)).to(device)
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
        mark('ddp_initialized')
        opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
        for _ in range(2):
            opt.zero_grad()
            loss = ddp(torch.randn(8, 16, device=device)).square().mean()
            loss.backward()
            opt.step()
        mark('passed', loss=loss.item())
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
