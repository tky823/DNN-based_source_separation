## Results
### 2 speakers
| FFT size | hop size | window | frequency mask | mask nonlinearity | K | H | B | causal | batch size | epoch | optimizer | lr (start/end) | lr scheduler | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 256 | 64 | Hann | IBM | sigmoid | 20 | 300 | 4 | False | 64 | 150 | RSMprop | 1e-4 / 3e-6 | exponential decay | None |  |  |  |

### 3 speakers
| FFT size | hop size | window | frequency mask | mask nonlinearity | K | H | B | causal | batch size | epoch | optimizer | lr (start/end) | lr scheduler | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 256 | 64 | Hann | IBM | sigmoid | 20 | 300 | 4 | False | 64 | 150 | RSMprop | 1e-4 / 3e-6 | exponential decay | None |  |  |  |