## Results
### 2 speakers
| encoder | decoder | mask nonlinearity | N | L | H | X | R | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainableGated | trainable | sigmoid | 500 | 40 | 500 | 2 | 2 | False | 128 | adam | 1e-3 | 5 | 11.4 | 11.8 | 2.84 |