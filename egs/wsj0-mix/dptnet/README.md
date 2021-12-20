## Results
We define `L` as a kernel size. 
### 2 speakers
| encoder | decoder | mask_nonlinear | N | L | F | K | P | B | d_ff | h | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | relu | 64 | 2 | 64 | 250 | 125 | 6 | 128 | 4 | False | 2 | adam | 1e-3 | 5 | 19.9 | 20.1 | 3.65 |
| trainable | trainable | relu | 64 | 2 | 64 | 250 | 125 | 6 | 128 | 4 | False | 4 | adam | 1e-3 | 5 | 19.7 | 19.9 | 3.63 |

### 3 speakers
| encoder | decoder | mask_nonlinear | N | L | F | K | P | B | d_ff | h | causal | batch size | optimizer | lr | gradient clipping | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| trainable | trainable | relu | 64 | 2 | 64 | 250 | 125 | 6 | 128 | 4 | False | 2 | adam | 1e-3 | 5 |  |  |  |
| trainable | trainable | relu | 64 | 2 | 64 | 250 | 125 | 6 | 128 | 4 | False | 4 | adam | 1e-3 | 5 | 15.6 | 15.9 | 2.23 |