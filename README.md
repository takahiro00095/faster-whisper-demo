## Overview
FasterWhisperの精度検証用リポジトリ

## 環境依存
- NVIDIA GPUの利用が前提となる
    - CUDA 12
    - cuDNN 9
- GPUがない場合でも一応動作はするが、低速となる

## ディレクトリ構成
```
/
├─ app/             アプリケーション本体
└─ evaluation/      精度検証用ユーティリティ
```

## QuickStart
### 動作環境
- Docker上で動作(DockerComposeの利用が前提)
- `localhost:8501`で動作する

### アプリケーションの起動
```bash
git clone https://github.com/takahiro00095/faster-whisper-demo
```
```bash
cd faster-whisper-demo/app
docker compose up
```
### 実行オプション
#### モデルサイズ
- `large-v3-turbo`推奨

#### 計算タイプ
- int8(量子化)にすると高速化するが精度は若干落ちる
- GPU利用の場合はfloat16を推奨。CPU利用の場合はint8が現実的

## 精度検証
`jiwer`を使った精度検証が可能（今のところ手作業）
`eval/out`などに検証用ファイルと正解ファイルを配置（テキストファイル）して、`eval.py`内のファイルパスを修正し、以下を実行
```bash
cd evaluation

# venvを有効化・依存性解決
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

# 精度検証実行
python3 eval.py 
```