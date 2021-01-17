## Results
### 2 speakers
| encoder | decoder | mask nonlinearity | N | L | H | B | Sc | P | X | R | causal | optimizer | lr | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 | 15.6 | 15.9 | 3.29 |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 |  |  |  |
| trainable | pseudo-inverse | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 |  |  |  |

### 3 speakers
| encoder | decoder | mask nonlinearity | N | L | H | B | Sc | P | X | R | causal | optimizer | lr | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 |  |  |  |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 |  |  |  |