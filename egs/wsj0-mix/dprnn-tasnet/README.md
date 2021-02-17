## Results
We define `L` as a kernel size. 
### 2 speakers
| encoder | decoder | mask nonlinearity | N | L | F | H | K | P | B | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | sigmoid | 64 | 2 | 64 | 128 | 250 | 125 | 6 | False | 2 | adam | 1e-3 | 5 |  |  |  |

### 3 speakers
| encoder | decoder | mask nonlinearity | N | L | F | H | K | P | B | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | sigmoid | 64 | 2 | 64 | 128 | 250 | 125 | 6 | False | 2 | adam | 1e-3 | 5 |  |  |  |