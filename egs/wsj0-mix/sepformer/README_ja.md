## 実験結果
### 2話者
| encoder | decoder | mask nonlinearity | F | L | B | C | P | R | K_intra | K_inter | h_intra | h_inter | d_ff_intra | d_ff_inter | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | ReLU | 256 | 16 | 256 | 250 | 125 | 2 | 8 | 8 | 8 | 8 | 1024 | 1024 | False | 128 | adam | 15e-5 | 5 | 18.1 | 18.3 | 3.44 |