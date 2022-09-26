# Pytorch MVDR example
Pytorch-lightningを使用したMVDRによる音声強調

## 実行環境構築
ターミナルを起動し、MVDRをクローンしたいディレクトリに移動し、  

    git clone https://github.com/Nushigawa03/MVDR.git
    cd MVDR
    conda env create -f environment.yml
    conda activate MVDR
    pip install ci-sdr
    pip install pytorch_lightning
と実行する

## データセット準備
https://zenodo.org/record/4642005  
クローンされたMVDRディレクトリの中にdataディレクトリを作成し、その中に上記URLからダウンロードしたL3DAS_Task1_dev.zipを解凍する  
デフォルトでは、MVDR\data\L3DAS_Task1_devとなるようにデータを配置することで学習ができる

## 実行方法
### train.py
デフォルトのパラメータで実行するには

    python train.py
より細かくパラメータを設定し実行するには

    python train.py -e [[エポック数]] -b [[バッチサイズ]] -d [[トレーニングデータのディレクトリ]] --every_n_steps [[学習のチェックポイントを保存するステップ数]] --model [[学習後出力されるモデルの名前]]
学習後、trainフォルダに学習モデル(model.ckpt)が生成される
### test.py
デフォルトのパラメータで実行するには

    python test.py
実行後、outフォルダにout/84-121123-0021_A.wavのノイズ除去した音声(output.wav)が生成される  
より細かくパラメータを設定し実行するには

    python test.py -i [[変換したい音声のパス]] -o [[出力される音声の名前]] --model [[使用するモデルのパス]]
