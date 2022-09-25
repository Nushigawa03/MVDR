# MVDR
Pytorch-lightningを使用したMVDRによる音声強調

## 実行環境構築
Anaconda Promptを起動し、MVDRをクローンしたいディレクトリに移動し、  

    git clone https://github.com/Nushigawa03/MVDR.git
    cd MVDR
    conda env create -f environment.yml
    conda activate MVDR
    pip install ci-sdr
    pip install pytorch_lightning
と実行する

## データセット準備
https://zenodo.org/record/4642005  
クローンされたMVDRフォルダの中にdataフォルダを作成しL3DAS_Task1_dev.zipを解凍する
デフォルトでは、MVDRフォルダ直下にMVDR\data\L3DAS_Task1_devとなるようにデータを配置することで学習ができる

## 実行方法
### train.py
trainフォルダに学習モデル(model.ckpt)が生成される

    python train.py
    python train.py -e [[エポック数]] -b [[バッチサイズ]] -d [[トレーニングデータのディレクトリ]] --every_n_steps [[学習のチェックポイントを保存するステップ数]] --model [[学習後出力されるモデルの名前]]
### test.py
outフォルダにノイズ除去した音声(output.wav)が生成される

    python test.py -i [[変換したい音声のパス]]
    python test.py -i [[変換したい音声のパス]] -o [[出力される音声の名前]] --model [[使用するモデルのパス]]
