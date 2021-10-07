## Results
### 2 speakers
| encoder | decoder | mask nonlinearity | N | L | H | B | Sc | P | X | R | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 15.5 | 15.8 | 3.27 |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 15.1 | 15.4 | 3.20 |
| trainable | pseudo-inverse | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 14.8 | 15.1 | 3.15 |
| Fourier | Fourier | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 14.9 | 15.2 | 3.11 |

### 3 speakers
| encoder | decoder | mask nonlinearity | N | L | H | B | Sc | P | X | R | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 11.3 | 11.7 | 1.88 |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 8 | adam | 1e-3 | 5 | 11.4 | 11.7 | 1.89 |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 11.2 | 11.6 | 1.86 |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 8 | adam | 1e-3 | 5 | 11.5 | 11.9 | 1.95 |