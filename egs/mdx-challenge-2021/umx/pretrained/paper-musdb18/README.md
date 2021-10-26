## How to do
1. Download pretrained models from [https://drive.google.com/file/d/1C67tgD79YIe-uEs31NTPMxuh7JNLPB7T/view?usp=sharing](https://drive.google.com/file/d/1C67tgD79YIe-uEs31NTPMxuh7JNLPB7T/view?usp=sharing).
2. Unzip `model.zip` under this directory.
3. Rename files as follows

```sh
model_choices=( "best" "last" )
sources=( "bass" "drums" "other" "vocals" )

for model_choice in "${model_choices[@]}" ; do
    if [ ! -d "${model_choice}" ] ; then
        mkdir "${model_choice}"
    fi
done

for model_choice in "${model_choices[@]}" ; do
    for s in "${sources[@]}" ; do
        mv "model/${s}/${model_choice}.pth" "${model_choice}/${s}.pth"
    done
done
```