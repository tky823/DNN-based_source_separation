# Example of outputs
I trained D3Net and share the outputs.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr/view?usp=sharing](https://drive.google.com/file/d/1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr/view?usp=sharing) that includes `bass/last.pth`, `bass/best.pth`, ...`vocals/best.pth`.
- You can download output JSON files from []().

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Vocals | Bass | Drums | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net | 44100 | - | - | - | - | - | - | Epoch is chosen by validation loss. |
| D3Net | 44100 | - | - | - | - | - | - | After 100 epochs. |