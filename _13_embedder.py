import torch


class DiscreteEmbbeder:
    def __init__(self, num_categories, bounds):
        self.num_categories = num_categories
        self.bounds = torch.tensor(bounds)
        self.range_size = torch.tensor(
            bounds[..., 1] - bounds[..., 0]) / num_categories
        midpoints = []
        for i in range(self.num_categories):
            midpoint = self.bounds[..., 0] + self.range_size * (i + 0.5)
            # rand uniform in range (- range_size/2, + range_size/2)
            rand = torch.rand(1) * self.range_size - self.range_size / 2
            midpoint = midpoint + rand

            # clip to bounds
            midpoint = torch.clamp(
                midpoint, min=self.bounds[..., 0], max=self.bounds[..., 1]
            )
            midpoints.append(midpoint.tolist())

        self.cat_range = torch.tensor(midpoints).T

    def encode(self, sentence, *args, **kwargs):
        """Cat2Con
        Args:
            sentence (torch.Tensor): A tensor of shape `... x num_categories` one
                hot vector
        """
        assert torch.any(sentence >= 0)
        assert torch.any(sentence < self.num_categories)
        return (self.cat_range * sentence).sum(dim=-1)

    def decode(self, sentence, *args, **kwargs):
        # Con2Cat
        assert torch.any(sentence >= self.bounds[..., 0])
        assert torch.any(sentence <= self.bounds[..., 1])
        cat = (sentence - self.bounds[..., 0]) / self.range_size
        cat = cat.long()
        cat[cat == self.num_categories] = self.num_categories - 1
        return cat

    def to(self, device="cpu", dtype=torch.float32):
        self.cat_range = self.cat_range.to(device=device, dtype=dtype)
        self.range_size = self.range_size.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return self
