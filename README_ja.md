Jananese/ [English](README.md)

# Re:ゼロから始めるChainer生活

## 概要

機械学習初心者がChainerを触ってみるためのサンプル。

## 使い方

    ruby makedata.rb  # 学習用データセット作成
    python train.py   # 学習 (test.modelが保存される)
    python test.py    # 学習できたか確認
    python export.py  # C++用にモデルをエクスポート
    make              # C++コードのコンパイル
    ./a.out           # モデルをインポートして確認

## データセット

* makedata.rb: 一次元データが上に凸か下に凸か分類
* evenodd.rb : 立っているビットが奇数個か偶数個か分類
