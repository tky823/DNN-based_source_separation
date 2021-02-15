## Results
We define `L` as a kernel size. 
### 2 speakers
| encoder | decoder | mask nonlinearity | N | L | F | H | K | P | B | causal | optimizer | lr | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | sigmoid | 64 | 2 | 64 | 128 | 250 | 125 | 6 | False | adam | 1e-3 |  |  |  |

### 3 speakers
| encoder | decoder | mask nonlinearity | N | L | F | H | K | P | B | causal | optimizer | lr | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | sigmoid | 64 | 2 | 64 | 128 | 250 | 125 | 6 | False | adam | 1e-3 |  |  |  |