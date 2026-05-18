import torch
import torch.nn as nn
import torchaudio.transforms as T


class FeatureAugmentation(nn.Module):
    def __init__(
        self,
        num_mel_bins: int,
        max_proportion: float,
        time_mask_param: int,
    ):
        super().__init__()
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param, p=max_proportion)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=int(max_proportion * num_mel_bins))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        is_2d = spec.ndim == 2
        if is_2d:
            spec = spec.unsqueeze(0)
        spec = spec.transpose(1, 2)

        spec = self.time_masking(spec)
        spec = self.freq_masking(spec)

        spec = spec.transpose(1, 2)
        if is_2d:
            spec = spec.squeeze(0)

        return spec
