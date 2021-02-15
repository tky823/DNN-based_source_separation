## 実験結果
### 2話者
| encoder | decoder | mask nonlinearity | N | L | H | B | Sc | P | X | R | causal | optimizer | lr | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 | 15.5 | 15.8 | 3.27 |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 | 15.1 | 15.4 | 3.20 |
| trainable | pseudo-inverse | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 | 14.9 | 15.1 | 3.16 |

### 3話者
| encoder | decoder | mask nonlinearity | N | L | H | B | Sc | P | X | R | causal | optimizer | lr | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable + ReLU | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 | 11.3 | 11.7 | 1.88 |
| trainable | trainable | sigmoid | 512 | 16 | 128 | 512 | 128 | 3 | 8 | 3 | False | adam | 1e-3 | 11.2 | 11.6 | 1.86 |