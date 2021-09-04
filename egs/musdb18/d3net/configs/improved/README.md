# Configuration example
Based on official implementation: https://github.com/sony/ai-research-code/tree/master/d3net/music-source-separation/configs

- Set `weight_decay=1e-4`.
- Set patch size
    - `patch=256`: for `drums` and `other`
    - `patch=352`: for `bass`
    - `patch=192`: for `vocals`