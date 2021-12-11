# Pretrained Models
| Model | Dataset | Example |
|:---:|:---:|:---:|
| DeepClustering | WSJ0-2mix | `model = DeepClustering.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| DeepClustering | WSJ0-3mix | `model = DeepClustering.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
| DANet | WSJ0-2mix | `model = DANet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| DANet | WSJ0-3mix | `model = DANet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
| DANet (fixed attractor) | WSJ0-2mix | `model = FixedAttractorDANet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| DANet (fixed attractor) | WSJ0-3mix | `model = FixedAttractorDANet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
| DANet | LibriSpeech | `model = DANet.build_from_pretrained(task="librispeech", sample_rate=16000, n_sources=2)` |
| ADANet | WSJ0-2mix | `model = ADANet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| ADANet | WSJ0-3mix | `model = ADANet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
| LSTM-TasNet | WSJ0-2mix | `model = LSTMTasNet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| LSTM-TasNet | WSJ0-3mix | `model = LSTMTasNet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
| Conv-TasNet | WSJ0-2mix | `model = ConvTasNet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| Conv-TasNet | WSJ0-3mix | `model = ConvTasNet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
| Conv-TasNet | MUSDB18 | `model = ConvTasNet.build_from_pretrained(task="musdb18", sample_rate=44100)` |
| Conv-TasNet | WHAM | `model = ConvTasNet.build_from_pretrained(task="wham/separate-noisy", sample_rate=8000)` |
| Conv-TasNet | WHAM | `model = ConvTasNet.build_from_pretrained(task="wham/enhance-single", sample_rate=8000)` |
| Conv-TasNet | WHAM | `model = ConvTasNet.build_from_pretrained(task="wham/enhance-both", sample_rate=8000)` |
| Conv-TasNet | LibriSpeech | `model = ConvTasNet.build_from_pretrained(task="librispeech", sample_rate=16000, n_sources=2)` |
| DPRNN-TasNet | WSJ0-2mix | `model = DPRNNTasNet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| DPRNN-TasNet | WSJ0-3mix | `model = DPRNNTasNet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
| DPRNN-TasNet | LibriSpeech | `model = DPRNNTasNet.build_from_pretrained(task="librispeech", sample_rate=16000, n_sources=2)` |
| MMDenseLSTM | MUSDB18 | `model = MMDenseLSTM.build_from_pretrained(task="musdb18", sample_rate=44100, target="vocals")` |
| MMDenseLSTM (bass, drums, other, vocals) | MUSDB18 | `model = ParallelMMDenseLSTM.build_from_pretrained(task="musdb18", sample_rate=44100)` |
| Open-Unmix | MUSDB18 | `model = OpenUnmix.build_from_pretrained(task="musdb18", sample_rate=44100, target="vocals")` |
| Open-Unmix (bass, drums, other, vocals) | MUSDB18 | `model = ParallelOpenUnmix.build_from_pretrained(task="musdb18", sample_rate=44100)` |
| Open-Unmix | MUSDB18-HQ | `model = OpenUnmix.build_from_pretrained(task="musdb18hq", sample_rate=44100, target="vocals")` |
| Open-Unmix (bass, drums, other, vocals) | MUSDB18-HQ | `model = ParallelOpenUnmix.build_from_pretrained(task="musdb18hq", sample_rate=44100)` |
| DPTNet | WSJ0-2mix | `model = DPTNet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| DPTNet | WSJ0-3mix | `model = DPTNet.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
| CrossNet-Open-Unmix | MUSDB18 | `model = CrossNetOpenUnmix.build_from_pretrained(task="musdb18", sample_rate=44100)` |
| D3Net | MUSDB18 | `model = D3Net.build_from_pretrained(task="musdb18", sample_rate=44100, target="vocals")` |
| D3Net (bass, drums, other, vocals) | MUSDB18 | `model = ParallelD3Net.build_from_pretrained(task="musdb18", sample_rate=44100)` |
| D3Net | MUSDB18-HQ | `model = D3Net.build_from_pretrained(task="musdb18hq", sample_rate=44100, target="vocals")` |
| D3Net (bass, drums, other, vocals) | MUSDB18-HQ | `model = ParallelD3Net.build_from_pretrained(task="musdb18hq", sample_rate=44100)` |
| SepFormer | WSJ0-2mix | `model = SepFormer.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=2)` |
| SepFormer | WSJ0-3mix | `model = SepFormer.build_from_pretrained(task="wsj0-mix", sample_rate=8000, n_sources=3)` |
