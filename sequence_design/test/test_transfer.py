import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend="nccl",  # 'nccl' is recommended for GPU communication
        init_method="tcp://127.0.0.1:23456",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    dist.destroy_process_group()


def broadcast_tensor(rank, world_size):
    setup(rank, world_size)

    # Only initialize the tensor on rank 0 (GPU 0)
    if rank == 0:
        tensor0 = torch.tensor([42.0], device="cuda:0")
        tensor1 = torch.tensor([43.0], device="cuda:0")
    else:
        tensor0 = torch.empty(1, device=f"cuda:{rank}")  # Allocate space on other GPUs
        tensor1 = torch.empty(1, device=f"cuda:{rank}")  # Allocate space on other GPUs

    print(f"Initialized tensor on rank {rank}")
    print(f"(Before) Rank {rank} has tensor0: {tensor0}")
    print(f"(Before) Rank {rank} has tensor1: {tensor1}")

    # Broadcasting tensor from rank 0 to all other ranks
    dist.broadcast(tensor0, src=0)

    print(f"Rank {rank} has tensor: {tensor0}")
    print(f"Rank {rank} has tensor: {tensor1}")

    cleanup()


if __name__ == "__main__":
    world_size = 2  # Using 2 GPUs (0 and 1)

    # Set up the process groups and run the code
    mp.spawn(broadcast_tensor, args=(world_size,), nprocs=world_size, join=True)
