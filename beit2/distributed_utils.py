import os
import torch.distributed as dist
from typing import Dict, Union


def is_sm_run():
    return "TRAINING_JOB_NAME" in os.environ


def initialize_process_group(
    setup_args: Dict[str, Union[int, str]], backend: str = "nccl"
) -> None:
    """
    Initialize process group.
    """
    master_addr, master_port = setup_args["master_addr"], setup_args["master_port"]
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["ID"] = str(setup_args["global_rank"])
    os.environ["LOCAL_RANK"] = str(setup_args["local_rank"])
    dist_url = f"tcp://{master_addr}:{master_port}"
    if backend == "hccl":
        dist.init_process_group(
            backend=backend,
            rank=setup_args["global_rank"],
            world_size=setup_args["world_size"],
        )
    elif backend == "smddp":
        os.environ["SMDATAPARALLEL_LMC_ENABLE"] = "1"
        import smdistributed.dataparallel.torch.torch_smddp
        dist.init_process_group(backend)
    else:
        dist.init_process_group(
            backend=backend,
            rank=setup_args["global_rank"],
            world_size=setup_args["world_size"],
            init_method=dist_url,
        )


def get_ddp_port(increment: int = 0) -> str:
    """
       PyTorch still may leave orphan processes in multi-gpu training.
       Therefore we use a deterministic way to obtain port,
       so that users are aware of orphan processes by seeing the port occupied.
       we add increment if we intentionally want to run several process groups in parallel.
       stolen from:
            https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py#L126
    Args:
        increment Optional(int): constant we will add to our deterministic port calculation

    Returns:
        port number will be used for ddp process group communication
    """
    default_port = 2**15 + 2**14 + hash(os.getuid()) % 2**14 + increment
    return str(default_port)


def get_setup_defaults(
    local_rank: int, processes_per_node: int
) -> Dict[str, Union[str, int]]:
    """
        get the default values for init_process_group func
    Args:
        local_rank: rank within current machine.
        processes_per_node: number of procs in this node

    Returns:
        Dict with default values.
    """
    world_size = get_num_nodes() * processes_per_node
    node_rank = get_node_rank()
    global_rank = (node_rank * processes_per_node) + local_rank
    if "MASTER_PORT" not in os.environ:
        default_port = get_ddp_port()
        os.environ["MASTER_PORT"] = default_port
    processes_group_setup_args = {
        "global_rank": global_rank,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "master_port": os.environ["MASTER_PORT"],
    }
    return processes_group_setup_args


def get_num_nodes() -> int:
    """
        return number of nodes in job cluster.
    Returns:
        integer
    """
    if is_sm_run():
        import json
        cluster_inf = json.loads(os.environ.get("SM_RESOURCE_CONFIG"))
        return len(cluster_inf["hosts"])
    return 1


def get_node_rank() -> int:
    """
        returns node rank
    Returns:
        int
    """
    if is_sm_run():
        import json
        cluster_inf = json.loads(os.environ.get("SM_RESOURCE_CONFIG"))
        return cluster_inf["hosts"].index(cluster_inf["current_host"])
    return 0
