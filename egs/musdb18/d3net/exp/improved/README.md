# Example of outputs
I trained D3Net and share the outputs.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr/view?usp=sharing](https://drive.google.com/file/d/1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr/view?usp=sharing) that includes `bass/last.pth`, `bass/best.pth`, ...`vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1OOKBfY0c08LodjKeWXA4ETGWMwzCSbdN/view?usp=sharing](https://drive.google.com/file/d/1OOKBfY0c08LodjKeWXA4ETGWMwzCSbdN/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net | 44100 | 4.98 | 6.40 | 4.40 | 7.06 | 13.42 | 5.71 | Epoch is chosen by validation loss. |
| D3Net | 44100 | 5.02 | 6.63 | 4.43 | 7.03 | 13.42 | 5.78 | After 100 epochs. |