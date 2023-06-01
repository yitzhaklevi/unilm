import argparse
import os
import torch
import torch.multiprocessing as mp
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
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    from habana_frameworks.torch.hpex import hmp
    import torch.distributed as dist
except Exception as e:
    HPU_AVAILABLE = False
    print('ERROR!!!! HABANA error: ', e)


lazy_ = True
mixed_precision_ = True
single_card_ = False
profile_ = False

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
    #hmp.convert(
    #    opt_level="O1",
    #    bf16_file_path=bf16_file_path,
    #    fp32_file_path=fp32_file_path,
    #    isVerbose=False,
    #)
    os.environ['LOWER_LIST'] = bf16_file_path
    os.environ['FP32_LIST'] = fp32_file_path

def get_model():
    #mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
    #                    'depth': 12, 'num_heads': 12, 'embed_dim': 768,
    #                    'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}
    mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
                        'depth': 48, 'num_heads': 16, 'embed_dim': 2048,
                        'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}
    model = BeitTrainingModule(mae_model_config=mae_model_config)
    return model


def train_main_loop(dl,
                curr_iter,
                skip_first_iters,
                n_stamps,
                time_stamps,
                stop,
                optimizer,
                log_every,
                local_rank,
                batch_size,
                device,
                model,
                loss,
                profiler):
    import time
    t0 = time.perf_counter()
    for batch_idx, batch_data in enumerate(dl, start=1):
        if curr_iter <= skip_first_iters+1:
            t0 = time.perf_counter()
        if len(time_stamps) == n_stamps:
            stop = True
            break
        optimizer.zero_grad(set_to_none=True)

        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                batch_data[k] = v.to(device, non_blocking=True)
        else:
            batch_data = batch_data.to(device, non_blocking=True)
        with torch.autocast(device_type='hpu', enabled=mixed_precision_, dtype=torch.bfloat16):
            outputs = model(batch_data)
            l = loss(outputs, batch_data)['loss']

        l.backward()
        if lazy_:
            htcore.mark_step()
        with hmp.disable_casts():
            optimizer.step()
        if lazy_:
            htcore.mark_step()
        if profiler is not None:
            profiler.step()

        if curr_iter > skip_first_iters and (curr_iter-skip_first_iters) % log_every == 0 and local_rank == 0:
            time_passed = time.perf_counter() - t0
            if not single_card_:
                samples_processed = dist.get_world_size() * batch_size * log_every
            else:
                samples_processed = 1 * batch_size * log_every
            time_stamps.append(samples_processed / (time.perf_counter() - t0))
            print('iteration ', curr_iter, ':')
            print(f'{samples_processed / time_passed} samples/second')
            t0 = time.perf_counter()
        curr_iter += 1
        if stop:
            break
    if local_rank == 0 and len(time_stamps)>1:
        from statistics import mean, variance
        print(time_stamps)
        print(f'samples rate: {mean(time_stamps)} samples/second\n'
                              f'variance: {variance(time_stamps)}')



def mp_fn(local_rank):
    if not single_card_:
        setup_args = get_setup_defaults(
            local_rank=local_rank, processes_per_node=hthpu.device_count()
        )
        initialize_process_group(setup_args=setup_args, backend='hccl')
    if mixed_precision_:
        enable_mixed_precision()
    device = torch.device("hpu")
    model = get_model()
    loss = NoDynamicShapesMaskedCrossedEntropy()
    from habana_frameworks.torch.hpex.optimizers import FusedAdamW
    optimizer = FusedAdamW(params=model.parameters(),lr=3e-4, weight_decay=0.05, eps=1e-8)
    #optimizer = torch.optim.AdamW(params=model.parameters(), amsgrad=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e9}B params')
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    print(f'memory required to fit model: {mem / 1e9} Gigabytes')
    if not lazy_:
        os.environ["PT_HPU_LAZY_MODE"] = "2"

    model.to(device)
    loss.to(device)
    if not single_card_:
        model = DDP(
            model,
            find_unused_parameters=True,
            broadcast_buffers=False,
            bucket_cap_mb=460,
        )
    batch_size = 104
    dl = DataLoader(BeitLocalDataset(batch_size * 10000), batch_size=batch_size, num_workers=10, pin_memory=True)
    model.train()
    log_every = 10
    n_stamps = 10
    skip_first_iters = 3
    curr_iter = 0
    time_stamps = []
    stop = False
    kargs = {
                'dl':dl,
                'curr_iter': curr_iter,
                'skip_first_iters': skip_first_iters,
                'n_stamps': n_stamps,
                'time_stamps': time_stamps,
                'stop': stop,
                'optimizer': optimizer,
                'log_every': log_every,
                'local_rank' : local_rank,
                'batch_size' : batch_size,
                'device' : device,
                'model' : model,
                'loss' : loss,
                'profiler' : None
    }
    if profile_:
        p_activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU]
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=20, active=5, repeat=1),
            activities=p_activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/unilm/TM_logs')) as profiler:

            kargs['profiler'] = profiler
            train_main_loop(**kargs)
    else:
        train_main_loop(**kargs)






def train_on_cpu():
    device = torch.device("cpu")
    model = get_model()
    loss = MaskedCrossedEntropy()
    optimizer = torch.optim.AdamW(params=model.parameters(), amsgrad=False)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e6}M params')
    batch_size = 120
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
        if not single_card_:
            mp.spawn(
                fn=mp_fn,
                nprocs=hthpu.device_count(),
                join=True,
            )
        else:
            mp_fn(0)
