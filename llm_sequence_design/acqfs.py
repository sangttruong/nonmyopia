import random


def acqf_random(dataset, num_samples, *args, **kwargs):
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), num_samples)
    return dataset.select(indices)
