# Example of outputs
I trained D3Net and share the outputs.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1We9ea5qe3Hhcw28w1XZl2KKogW9wdzKF/view?usp=sharing](https://drive.google.com/file/d/1We9ea5qe3Hhcw28w1XZl2KKogW9wdzKF/view?usp=sharing) that includes `bass/last.pth`, `bass/best.pth`, ...`vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1hrhUekc-BJbVeZw6xdTPU2Gg3Dael5vM/view?usp=sharing](https://drive.google.com/file/d/1hrhUekc-BJbVeZw6xdTPU2Gg3Dael5vM/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net | 44100 | 5.26 | 6.34 | 4.54 | 6.88 | 13.35 | 5.76 | Epoch is chosen by validation loss. |
| D3Net | 44100 | 5.27 | 6.38 | 4.57 | 6.89 | 13.34 | 5.77 | After 100 epochs. |