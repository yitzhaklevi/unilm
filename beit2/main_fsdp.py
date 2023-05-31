import torch
from torch.utils.data import DataLoader
from beit2.beit_model import BeitTrainingModule
from beit2.beit_local_dataset import BeitLocalDataset
from beit2.beit_v2_loss import NoDynamicShapesCrossedEntropy
from beit2.distributed_utils import get_setup_defaults, initialize_process_group

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from timm.models.vision_transformer import Block
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)
patch_size = 16
image_key = "img"
img_shape = (224, 224)
NH = img_shape[0]//patch_size
NW = img_shape[1]//patch_size


def get_model():
    mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
                        'depth': 48, 'num_heads': 16, 'embed_dim': 2048,
                        'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}
    model = BeitTrainingModule(mae_model_config=mae_model_config)
    return model


def mp_fn(local_rank):
    setup_args = get_setup_defaults(
        local_rank=local_rank, processes_per_node=torch.cuda.device_count()
    )
    initialize_process_group(setup_args=setup_args, backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    model = get_model()
    num_params_before_shard = sum(p.numel() for p in model.parameters())

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )
    from functools import partial
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
        mixed_precision=mixed_precision,
        forward_prefetch=True,
        ignored_modules=[model.tokenizer],
        sync_module_states=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank
    )
    checkpointing_policy = partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpointing_policy,
        check_fn=lambda layer: isinstance(layer, Block),
    )

    loss = NoDynamicShapesCrossedEntropy()
    optimizer = torch.optim.AdamW(params=model.parameters(), amsgrad=False)
    num_params_after_shard = sum(p.numel() for p in model.parameters())
    print(f' model with {num_params_before_shard / 1e6}M params,'
          f' after shard remaining with {num_params_after_shard / 1e6}M params')

    batch_size = 128
    dl = DataLoader(BeitLocalDataset(batch_size * 100), batch_size=batch_size, num_workers=10)
    import time
    model.train()
    log_interval = 10
    t0 = time.perf_counter()

    model.to(device)
    loss.to(device)
    for batch_idx, batch_data in enumerate(dl, start=1):
        break
    for batch_idx in range(10000):
        optimizer.zero_grad(set_to_none=True)

        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                batch_data[k] = v.to(device)
        else:
            batch_data = batch_data.to(device)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(batch_data)
            l = loss(outputs, batch_data)['loss']

        l.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and local_rank == 0:
            time_passed = time.perf_counter() - t0
            samples_processed = dist.get_world_size() * batch_size * log_interval
            print(f'{samples_processed / time_passed} samples/second')
            t0 = time.perf_counter()
    print(f'{local_rank} done.')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    nprocs = torch.cuda.device_count()
    mp.spawn(mp_fn, args=(), nprocs=torch.cuda.device_count())
