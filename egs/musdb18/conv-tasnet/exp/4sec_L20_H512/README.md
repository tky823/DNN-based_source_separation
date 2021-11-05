# Example of outputs
I trained Conv-TasNet and share the outputs. See `config/paper/augmentation.yaml` for augmentation.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1A6dIofHZJQCUkyq-vxZ6KbPmEHLcf4WK/view?usp=sharing](https://drive.google.com/file/d/1A6dIofHZJQCUkyq-vxZ6KbPmEHLcf4WK/view?usp=sharing) that includes `last.pth` and `best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1KDjKJsiVex0D2P1Q4q--6CXo514_x2xA/view?usp=sharing](https://drive.google.com/file/d/1KDjKJsiVex0D2P1Q4q--6CXo514_x2xA/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 4 | 20 | 256 | 512 | 5.32 | 6.06 | 4.00 | 6.04 | 12.33 | 5.35 | Epoch is chosen by validation loss. |
| Conv-TasNet | 44100 | 4 | 20 | 256 | 512 | 4.82 | 5.98 | 3.73 | 6.06 | 12.32 | 5.15 | After 100 epochs. |