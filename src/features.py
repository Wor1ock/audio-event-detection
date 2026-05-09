import torch
import torch.nn as nn
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor


class FeatureAugmentation(nn.Module):
    def __init__(self, max_proportion: float = 0.3):
        super().__init__()
        self.time_masking = T.TimeMasking(time_mask_param=192, p=max_proportion)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=int(max_proportion * 128))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # ASTExtractor имеет форму [Batch, Time, Freq] -> [1, 1024, 128]
        # Torchaudio Masking ожидает [..., Freq, Time]
        spec = spec.transpose(1, 2)
        spec = self.freq_masking(self.time_masking(spec))
        return spec.transpose(1, 2)


class AudioFeatureExtractor(nn.Module):
    def __init__(self, sample_rate: int = 16000, need_augment: bool = False, max_proportion: float = 0.3):
        super().__init__()
        self.sample_rate = sample_rate
        self.need_augment = need_augment

        self.extractor = ASTFeatureExtractor.from_pretrained(
            "/home/ext-yankin@ad.speechpro.com/Документы/labs/audio-event-detection/models/ast_base"
        )

        if self.need_augment:
            self.augmentation = FeatureAugmentation(max_proportion=max_proportion)

    def forward(self, waveform: torch.Tensor, force_augment: bool | None = None) -> torch.Tensor:
        inputs = self.extractor(
            waveform.squeeze().cpu().numpy(), sampling_rate=self.sample_rate, padding="max_length", return_tensors="pt"
        )
        feature = inputs.input_values  # [1, 1024, 128]

        do_aug = force_augment if force_augment is not None else self.need_augment
        if do_aug:
            feature = self.augmentation(feature)

        return feature  # [1, 1024, 128]
