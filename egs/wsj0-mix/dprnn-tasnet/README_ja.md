## 実験結果
`L` はカーネルサイズを表している．
### 2話者
| encoder | decoder | mask nonlinearity | N | L | F | H | K | P | B | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | sigmoid | 64 | 2 | 64 | 128 | 250 | 125 | 6 | False | 2 | adam | 1e-3 | 5 | 18.6 | 18.8 | 3.54 |

### 3話者
| encoder | decoder | mask nonlinearity | N | L | F | H | K | P | B | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | sigmoid | 64 | 2 | 64 | 128 | 250 | 125 | 6 | False | 2 | adam | 1e-3 | 5 |  |  |  |