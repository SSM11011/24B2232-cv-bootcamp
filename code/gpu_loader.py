"""Keeps the whole dataset on the GPU and augments there.

Same idea as the GPU loader in Keller Jordan's airbench: upload the uint8
images once, then build each batch with tensor indexing instead of a CPU
DataLoader. Augmentation is the standard CIFAR recipe: random 32x32 crop from
the image zero-padded by 4 pixels, then a horizontal flip with probability 0.5.
"""
import torch
import torch.nn.functional as F

PAD = 4
SIZE = 32


class GpuImages:
    """A fixed set of images living on the GPU as uint8, NCHW."""

    def __init__(self, images_nhwc, mean, std, device="cuda"):
        self.images = torch.from_numpy(images_nhwc).to(device).permute(0, 3, 1, 2).contiguous()
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self._padded = None

    def __len__(self):
        return len(self.images)

    def normalize(self, batch_uint8):
        x = (batch_uint8.float() / 255 - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)

    def plain_batches(self, batch_size):
        """Un-augmented batches in order, for evaluation."""
        for start in range(0, len(self), batch_size):
            yield self.normalize(self.images[start:start + batch_size])

    def augmented_batch(self, idx, generator):
        """Random crop + flip of the images at positions `idx`."""
        if self._padded is None:
            self._padded = F.pad(self.images, (PAD, PAD, PAD, PAD))  # zero padding, padded once
        n, device = len(idx), idx.device

        # Pick a 32x32 window inside each 40x40 padded image.
        top = torch.randint(0, 2 * PAD + 1, (n,), device=device, generator=generator)
        left = torch.randint(0, 2 * PAD + 1, (n,), device=device, generator=generator)
        offsets = torch.arange(SIZE, device=device)
        rows = (top[:, None] + offsets)[:, None, :, None]    # (n, 1, 32, 1)
        cols = (left[:, None] + offsets)[:, None, None, :]   # (n, 1, 1, 32)
        channels = torch.arange(3, device=device)[None, :, None, None]
        batch = self._padded[idx[:, None, None, None], channels, rows, cols]

        flip = torch.rand(n, device=device, generator=generator) < 0.5
        batch = torch.where(flip[:, None, None, None], batch.flip(3), batch)
        return self.normalize(batch)
