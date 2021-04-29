# Conv-TasNet using ORPIT
Reference: [Recursive speech separation for unknown number of speakers](https://arxiv.org/abs/1904.03065)

## Preparation of dataset
```shell
cd ../common

. ./prepare_unknown_number_sources.sh \
--mixed_n_sources '2+3' \
--wsj0mix_root <WSJ0MIX_ROOT>
```

## Training
```shell
. ./train.sh --exp_dir <OUTPUT_DIR>
```

## Resume Training
```shell
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <MODEL_PATH>
```

## Finetuning
```shell
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <MODEL_PATH>
```

## Resume Finetuning
```shell
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <FINETUNED_MODEL_PATH>
```