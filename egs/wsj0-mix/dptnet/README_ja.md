## 実験結果
`L` はカーネルサイズを表している．
### 2 speakers
| encoder | decoder | mask_nonlinear | N | L | F | K | P | B | d_ff | h | causal | optimizer | lr | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | relu | 64 | 2 | 64 | 250 | 125 | 6 | 128 | 4 | False | adam | 1e-3 |  |  |  |

### 3 speakers
| encoder | decoder | mask_nonlinear | N | L | F | K | P | B | d_ff | h | causal | optimizer | lr | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | relu | 64 | 2 | 64 | 250 | 125 | 6 | 128 | 4 | False | adam | 1e-3 |  |  |  |