import argparse
import os
import torch
from torch.utils.data import DataLoader
from beit2.beit_model import BeitTrainingModule
from beit2.beit_local_dataset import BeitLocalDataset
from beit2.beit_v2_loss import NoDynamicShapesMaskedCrossedEntropy, MaskedCrossedEntropy
from beit2.distributed_utils import get_setup_defaults, initialize_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import chain
from typing import Dict, Union
import argparse
HPU_AVAILABLE = True
try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    from habana_frameworks.torch.hpex import hmp
    import torch.distributed as dist
except:
    HPU_AVAILABLE = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cpu', 'hpu'], required=True)
    return parser.parse_args()


patch_size = 16
image_key = "img"
img_shape = (224, 224)
NH = img_shape[0]//patch_size
NW = img_shape[1]//patch_size


def enable_mixed_precision():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    bf16_file_path = os.path.join(dir_path, "ops_bf16.txt")
    fp32_file_path = os.path.join(dir_path, "ops_fp32.txt")
    hmp.convert(
        opt_level="O1",
        bf16_file_path=bf16_file_path,
        fp32_file_path=fp32_file_path,
        isVerbose=False,
    )

def get_model():
    mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
                        'depth': 12, 'num_heads': 12, 'embed_dim': 768,
                        'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}
    model = BeitTrainingModule(mae_model_config=mae_model_config)
    return model


def mp_fn(local_rank):
    setup_args = get_setup_defaults(
        local_rank=local_rank, processes_per_node=hthpu.device_count()
    )
    initialize_process_group(setup_args=setup_args, backend='hccl')
    enable_mixed_precision()
    device = torch.device("hpu")
    model = get_model()
    loss = NoDynamicShapesMaskedCrossedEntropy()
    optimizer = torch.optim.AdamW(params=model.parameters(), amsgrad=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e6}M params')
    model = DDP(
        model,
        find_unused_parameters=True,
        broadcast_buffers=False,
        bucket_cap_mb=460,
    )
    batch_size = 128
    dl = DataLoader(BeitLocalDataset(batch_size * 10000), batch_size=batch_size, num_workers=10)
    import time
    model.train()
    log_interval = 10
    t0 = time.perf_counter()

    model.to(device)
    loss.to(device)
    for batch_idx, batch_data in enumerate(dl, start=1):
        optimizer.zero_grad(set_to_none=True)

        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                batch_data[k] = v.to(device)
        else:
            batch_data = batch_data.to(device)
        with torch.autocast(device_type='hpu', enabled=True, dtype=torch.bfloat16):
            outputs = model(batch_data)
            l = loss(outputs, batch_data)['loss']

        l.backward()
        htcore.mark_step()
        with hmp.disable_casts():
            optimizer.step()
            htcore.mark_step()

        if batch_idx % log_interval == 0 and local_rank == 0:
            time_passed = time.perf_counter() - t0
            samples_processed = dist.get_world_size() * batch_size * log_interval
            print(f'{samples_processed / time_passed} samples/second')
            t0 = time.perf_counter()



def train_on_cpu():
    device = torch.device("cpu")
    model = get_model()
    loss = MaskedCrossedEntropy()
    optimizer = torch.optim.AdamW(params=model.parameters(), amsgrad=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e6}M params')
    batch_size = 128
    dl = DataLoader(BeitLocalDataset(batch_size * 10000), batch_size=batch_size, num_workers=10)
    import time
    model.train()
    log_interval = 10
    t0 = time.perf_counter()

    model.to(device)
    loss.to(device)
    for batch_idx, batch_data in enumerate(dl, start=1):
        optimizer.zero_grad(set_to_none=True)

        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                batch_data[k] = v.to(device)
        else:
            batch_data = batch_data.to(device)
        outputs = model(batch_data)
        l = loss(outputs, batch_data)['loss']
        l.backward()


        if batch_idx % log_interval:
            time_passed = time.perf_counter() - t0
            samples_processed = batch_size * log_interval
            print(f'{samples_processed / time_passed} samples/second')
            t0 = time.perf_counter()


if __name__ == '__main__':
    args = get_args()
    if args.device == 'cpu':
        train_on_cpu()
    else:
        assert (
            hthpu.is_available() and HPU_AVAILABLE
        ), "hpu training only supported on machines with habana gaudi accelerator"

