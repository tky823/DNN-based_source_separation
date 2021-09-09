# Example of outputs
I trained D3Net and share the outputs. The networks are all trained by default setting except for random scaling. See `config/improved/augmentation-*.yaml`.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1UXtrJ0eo__Gonwqe9FoJaPQY4kFg_6aE/view?usp=sharing](https://drive.google.com/file/d/1UXtrJ0eo__Gonwqe9FoJaPQY4kFg_6aE/view?usp=sharing), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1GU2CBgHKrRcFJZHzwckcdHAgVA7xqRGx/view?usp=sharing](https://drive.google.com/file/d/1GU2CBgHKrRcFJZHzwckcdHAgVA7xqRGx/view?usp=sharing).

## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net w/o dilation | - | - | - | - | - | - | - |
| D3Net standard dilation | - | - | - | - | - | - | - |
| D3Net | 6.58 | 6.46 | 5.12 | 4.54 | 13.06 | 5.68 | Epoch is chosen by validation loss. |
| D3Net | 6.63 | 6.40 | 5.24 | 4.58 | 13.24 | 5.71 | After 50 epochs. |