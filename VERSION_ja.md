# バージョン情報
## v0.0.0
- Conv-TasNetとDPRNN-TasNetによるLibriSpeechデータセットの例を含む．

## v0.0.1
- データセット名の変更．

## v0.1.0
- データセットの構造の変更．

## v0.1.1
- DANetを追加．

## v0.1.2
- Conv-TasNetのレイヤー名の変更．DANetの入力特徴量をlog-magnitudeに変更．

## v0.1.3
- Wall Street Journal 0 (WSJ0) データセット用のスクリプトを追加．

## v0.1.4
- 非負値行列因子分解 (non-nagative matrix factorization; NMF)を追加．

## v0.2.0
- 短時間フーリエ変換の表現を変更．

## v0.2.1
- `conv_tasnet`ディレクトリの名前を`conv-tasnet`に変更．ORPIT (one-and-rest PIT)を追加．

## v0.3.0
- `wsj0`を`wsj0-mix`へ名前を変更．実験結果の更新．

## v0.3.1
- TasNetにおける線形のencoderに対応．

## v0.3.2
- dual-path RNNのチャネル数の定義を変更．

## v0.3.3
- v0.3.2の影響により，学習済みモデルを更新．

## v0.4.0
- DPRNN-TasNetのネットワーク構造を変更．

## v0.4.1
- DPTNetおよびGALRNetを追加．DPRNN-TasNetを再修正．

## v0.4.2
- GALRNet用の学習スクリプトを追加．

## v0.4.3
- DPRNN-TasNetを再修正．

## v0.5.0
- `parse_options.sh`を追加．

## v0.5.1
- 一部のモデルで多チャネルの入力に対応．

## v0.5.2
- 距離学習に関するチュートリアルを追加．

## v0.5.3
- D3Netの構造を修正．

## v0.5.4
- D3Netの構造を修正．

## v0.5.5
- MUSDB18データセットをwavで読み込むように変更．
- D3Netの学習と推論・評価のスクリプトを追加．
- D3NetをMUSDB18データセットで学習した結果を追加．

# v0.6.0
- D3Netのバグを修正．

# v0.6.1
- モジュールの追加．
  - LaSAFTに関するモジュールの追加．
- WHAMデータセットの追加．

# v0.6.2
- 学習受みモデルのリンクを修正．
- 一部のモデルのattribute名を変更．

# v0.6.3
- 結果の更新．
- MDXチャレンジ2021の例を追加.

# v0.7.0
- モデルの追加（`MMDenseLSTM`，`X-UMX`，`HRNet`，`SepFormer`）．
- 学習済みモデルの追加．

# v0.7.1
- モデルの追加（`DeepClustering`，`ADANet`）．
- モデルの修正（`LSTM-TasNet`）．
- パラメータ名の変更（`fft_size`->`n_fft`，`hop_size`->`hop_length`）．

# v0.7.2
- Jupyter notebookの更新．