# Example of outputs
I trained Conv-TasNet and share the outputs. See `4sec_L20_H512/config/augmentation-*.yaml` for augmentation.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1a-IQn3hsN84N2X_WF84uXbwe7VAk_vaB/view?usp=sharing](https://drive.google.com/file/d/1a-IQn3hsN84N2X_WF84uXbwe7VAk_vaB/view?usp=sharing) that includes `last.pth` and `best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1KDjKJsiVex0D2P1Q4q--6CXo514_x2xA/view?usp=sharing](https://drive.google.com/file/d/1KDjKJsiVex0D2P1Q4q--6CXo514_x2xA/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 4 | 20 | 256 | 512 | 6.04 | 6.06 | 5.32 | 4.00 | 12.33 | 5.35 | Epoch is chosen by validation loss. |
| Conv-TasNet | 44100 | 4 | 20 | 256 | 512 | 6.06 | 5.98 | 4.82 | 3.73 | 12.32 | 5.15 | After 100 epochs. |