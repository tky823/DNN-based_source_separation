from mir_eval.separation import bss_eval_sources as bss_eval_sources_np
import torch

def bss_eval_sources(reference_sources: torch.Tensor, estimated_sources: torch.Tensor, **kwargs):
    """
    Wrapper of mir_eval.separation.bss_eval_sources
    Args:
        reference_sources <torch.Tensor>: (n_sources, T)
    Returns:
        sdr <torch.DoubleTensor>: (n_sources,)
        sir <torch.DoubleTensor>: (n_sources,)
        sar <torch.DoubleTensor>: (n_sources,)
        perm <torch.LongTensor>: (n_sources,)
    """
    reference_sources = reference_sources.numpy()
    estimated_sources = estimated_sources.numpy()

    result = bss_eval_sources_np(
        reference_sources=reference_sources,
        estimated_sources=estimated_sources,
        **kwargs
    )

    result = [
        torch.from_numpy(_result) for _result in result
    ]

    sdr, sir, sar, perm = result

    return sdr, sir, sar, perm

def _test_bss_eval_sources():
    reference_source_man, _ = torchaudio.load("data/single-channel/man-16000.wav")
    reference_source_woman, _ = torchaudio.load("data/single-channel/woman-16000.wav")
    estimated_source_man, _ = torchaudio.load("data/single-channel/man-oracle-16000.wav")
    estimated_source_woman, _ = torchaudio.load("data/single-channel/woman-16000.wav")

    reference_sources = torch.cat([reference_source_man, reference_source_woman], dim=0)
    estimated_sources = torch.cat([estimated_source_man, estimated_source_woman], dim=0)

    result = bss_eval_sources(reference_sources=reference_sources, estimated_sources=estimated_sources)

    print(result)

if __name__ == '__main__':
    import torchaudio

    _test_bss_eval_sources()