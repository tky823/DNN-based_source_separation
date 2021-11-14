# Example of outputs
I trained Conv-TasNet and share the outputs. See `config/paper/augmentation.yaml` for augmentation.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1paXNGgH8m0kiJTQnn1WH-jEIurCKXwtw/view?usp=sharing](https://drive.google.com/file/d/1paXNGgH8m0kiJTQnn1WH-jEIurCKXwtw/view?usp=sharing) that includes `last.pth` and `best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1R-63xiocr6Cjp84O9pKej-X2iZFb-QOl/view?usp=sharing](https://drive.google.com/file/d/1R-63xiocr6Cjp84O9pKej-X2iZFb-QOl/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 8 | 64 | 256 | 512 | 5.38 | 5.82 | 3.51 | 5.91 | 11.85 | 5.16 | Epoch is chosen by validation loss. |
| Conv-TasNet | 44100 | 8 | 64 | 256 | 512 | 5.33 | 5.79 | 3.48 | 6.02 | 11.91 | 5.16 | After 100 epochs. |