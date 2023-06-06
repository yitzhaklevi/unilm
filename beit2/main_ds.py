import argparse
import os
import torch
from torch.utils.data import DataLoader
from beit2.beit_model import BeitTrainingModule
from beit2.beit_local_dataset import BeitLocalDataset
from beit2.beit_v2_loss import NoDynamicShapesMaskedCrossedEntropy, MaskedCrossedEntropy
from itertools import chain
from typing import Dict, Union
import argparse
import deepspeed
HPU_AVAILABLE = True
try:
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    import habana_frameworks.torch.distributed.hccl
except Exception as e:
    HPU_AVAILABLE = False
    print('ERROR!!!! HABANA error: ', e)


profile_ = False
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cpu', 'hpu'], required=True)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args(), parser.parse_known_args()[0]


patch_size = 16
image_key = "img"
img_shape = (224, 224)
NH = img_shape[0]//patch_size
NW = img_shape[1]//patch_size

def get_model():
    #mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
    #                    'depth': 12, 'num_heads': 12, 'embed_dim': 768,
    #                    'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}

    #2.5B params
    #mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
    #                    'depth': 48, 'num_heads': 16, 'embed_dim': 2048,
    #                    'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}
    #4B params vit-e mlp_ratio = 15360/1792
    mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
            'depth': 56, 'num_heads': 16, 'embed_dim': 1792, 'mlp_ratio': float(15360/1792),
                        'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}
    #22B params mlp_ratio = 24573/6144
#    mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
#                        'depth': 48, 'num_heads': 48, 'embed_dim': 6144, 'mlp_ratio': float(24573/6144) 
#                        'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}

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
                world_size,
                batch_size,
                device,
                model_engine,
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
        outputs = model_enigne(batch_data)
        l = loss(outputs, batch_data)['loss']

        model_engine.backward(l)
        
        htcore.mark_step()

        model_engine.step()

        htcore.mark_step()

        if profiler is not None:
            profiler.step()

        if curr_iter > skip_first_iters and (curr_iter-skip_first_iters) % log_every == 0 and local_rank == 0:
            time_passed = time.perf_counter() - t0
            samples_processed = world_size * batch_size * log_every
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



def mp_fn(deepSpeed_config):
    os.environ["MAX_WAIT_ATTEMPTS"] = "50"

    deepspeed.init_distributed(dist_backend='hccl')
    device = torch.device("hpu")
    model = get_model()
    loss = NoDynamicShapesMaskedCrossedEntropy()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    optimizer_kwargs = {'params': optimizer_grouped_parameters, 'lr': 3e-4, 'weight_decay': 0.05, 'eps': 1e-8}

    from habana_frameworks.torch.hpex.optimizers import FusedAdamW
    optimizer =  FusedAdamW(**optimizer_kwargs)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e9}B params')
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs
    print(f'memory required to fit model: {mem / 1e9} Gigabytes')

    model.to(device, dtype=torch.bfloat16)
    loss.to(device, dtype=torch.bfloat16)
    batch_size = 64
    dl = DataLoader(BeitLocalDataset(batch_size * 10000), batch_size=batch_size, num_workers=10, pin_memory=True)
    model.train()
    model_engine, optimizer ,train_loader,_ = deepspeed.initialize(args=deepSpeed_config, model=model, optimizer=optimizer ,model_parameters=None, training_data=dl.dataset)
    _, train_micro_batch_size_per_gpu, _ = model.get_batch_info()
    ds_batch_size = train_micro_batch_size_per_gpu()
    if ds_batch_size != batch_size:
        raise(BaseException("deep speed batch size diffrenet from script batch size, "+str(ds_batch_size)+"!="+str(batch_size)))
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
                'local_rank' : int(os.environ["RANK"]),
                'batch_size' : batch_size,
                'device' : device,
                'model_engine' : model_engine,
                'loss' : loss,
                'profiler' : None,
                'world_size' : int(os.environ["WORLD_SIZE"])
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
    args, deepspeed_config = get_args()
    if args.device == 'cpu':
        train_on_cpu()
    else:
        assert (
            hthpu.is_available() and HPU_AVAILABLE
        ), "hpu training only supported on machines with habana gaudi accelerator"
    mp_fn(deepspeed_config)
