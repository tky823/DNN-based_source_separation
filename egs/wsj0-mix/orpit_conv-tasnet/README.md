# Conv-TasNet using ORPIT
Reference: [Recursive speech separation for unknown number of speakers](https://arxiv.org/abs/1904.03065)

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