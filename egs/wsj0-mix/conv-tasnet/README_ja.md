## 実験結果
### 2話者
| encoder | decoder | mask nonlinearity | N | L | H | B | Sc | P | X | R | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 15.5 | 15.8 | 3.27 |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 15.1 | 15.4 | 3.20 |
| trainable | pseudo-inverse | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 14.9 | 15.1 | 3.16 |

### 3話者
| encoder | decoder | mask nonlinearity | N | L | H | B | Sc | P | X | R | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 11.3 | 11.7 | 1.88 |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | 4 | adam | 1e-3 | 5 | 11.2 | 11.6 | 1.86 |