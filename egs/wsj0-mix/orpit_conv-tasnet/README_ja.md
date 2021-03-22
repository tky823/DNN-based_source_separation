# Conv-TasNet using ORPIT
Reference: [Recursive speech separation for unknown number of speakers](https://arxiv.org/abs/1904.03065)

## データセットの準備
```shell
cd ../common

. ./prepare_unknown_number_sources.sh \
--mixed_n_sources '2+3' \
--wsj0mix_root <WSJ0MIX_ROOT>
```

## 学習
```shell
. ./train.sh --exp_dir <OUTPUT_DIR>
```

## 学習の再開
```shell
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <MODEL_PATH>
```

## Finetuning
```shell
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <MODEL_PATH>
```

## Finetuningの再開
```shell
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <FINETUNED_MODEL_PATH>
```