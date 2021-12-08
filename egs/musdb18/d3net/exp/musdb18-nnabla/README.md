# Example of outputs
I trained D3Net and share the outputs.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1B4e4e-8-T1oKzSg8WJ8RIbZ99QASamPB/view?usp=sharing](https://drive.google.com/file/d/1B4e4e-8-T1oKzSg8WJ8RIbZ99QASamPB/view?usp=sharing) that includes `bass/last.pth`, `bass/best.pth`, ...`vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1BYhKXMdnJL7RD71Gy9DWjK-u1XU08CKZ/view?usp=sharing](https://drive.google.com/file/d/1BYhKXMdnJL7RD71Gy9DWjK-u1XU08CKZ/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net | 44100 | 5.24 | 6.71 | 4.59 | 6.97 | 13.22 | 5.88 | Epoch is chosen by validation loss. |
| D3Net | 44100 | 5.24 | 6.65 | 4.52 | 6.91 | 13.29 | 5.83 | After 100 epochs. |