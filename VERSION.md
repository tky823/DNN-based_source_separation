# Version information
## v0.0.0
- Initial version. LibriSpeech Conv-TasNet & DPRNN-TasNet examples are included.

## v0.0.1
- Dataset is renamed.

## v0.1.0
- Dataset structure is changed.

## v0.1.1
- DANet is included.

## v0.1.2
- Layer name is changed. Input feature for DANet is replaced by log-magnitude.

## v0.1.3
- Add scripts for Wall Street Journal 0 (WSJ0) dataset.

## v0.1.4
- Add non-nagative matrix factorization (NMF).

## v0.2.0
- Change the representation of short time Fourier transform (STFT).

## v0.2.1
- `conv_tasnet` directory is renamed to `conv-tasnet`. Add one-and-rest PIT (ORPIT).

## v0.3.0
- `wsj0` is renamed to `wsj0-mix`. The result is updated.

## v0.3.1
- Implement Linear encoder for TasNet.

## v0.3.2
- Change the definition of `hidden_channels` in dual-path RNN.

## v0.3.3
- Fix trained models due to the update v0.3.2.

## v0.4.0
- Fix the network architecture of DPRNN-TasNet.

## v0.4.1
- Add DPTNet and GALRNet. Re-fix DPRNN-TasNet.

## v0.4.2
- Add training script for GALRNet.

## v0.4.3
- Re-fix DPRNN-TasNet.

## v0.5.0
- Add `parse_options.sh`.

## v0.5.1
- Multichannel support.

## v0.5.2
- Add metric learning tutorials.

## v0.5.3
- Update network architecture of D3Net.

## v0.5.4
- Bug fixes of D3Net.

## v0.5.5
- Load audio as wav for MUSDB18.
- Add training, estimation, and evaluation scripts for D3Net.
- Add results of D3Net using MUSDB18 dataset.

## v0.6.0
- Bug fixes of D3Net.

# v0.6.1
- Add modules.
  - LaSAFT related modules.
- Add WHAM dataset.

# v0.6.2
- Modify links to pretrained models.
- Change attibute name in some model.

# v0.6.3
- Update results.
- Add MDX challenge 2021 examples.

# v0.7.0
- Add new models (`MMDenseLSTM`, `X-UMX`, `HRNet`, `SepFormer`).
- Add pretrained models.

# v0.7.1
- Add new models (`DeepClustering`, `ADANet`).
- Bug fixes of (`LSTM-TasNet`).
- Rename parameters (`fft_size`->`n_fft`, `hop_size`->`hop_length`).

# v0.7.2
- Update jupyter notebooks.