# Auto-Generating-Inputs-For-Smoothed-Analysis-In-Message-Passing-Model

HiSampler
- https://github.com/joisino/HiSampler

python 3.8.0
- https://qiita.com/chem-it-village/items/7c55881bcd4ca33e6327#python%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB
- https://qiita.com/syusuke9999/items/9d35bcdb4119a1c4957a

C++の環境構築
- https://www.javadrive.jp/cstart/install/index6.html
- https://www.binarydevelop.com/article/libgcc_s_dw21dll-36074

pip install
- numpy 1.23.0
- chainer 7.8.1
- matplotlib 3.7.5
- pyyaml 6.0.2

make
- https://qiita.com/ryuso/items/cf0c1d83544103cacf05

gpuの設定
- https://www.kkaneko.jp/tools/win/chainer.html
- yamlファイルのgpuを0に変更

ファイルの変更
- util.py 63line　-> conf = yaml.safe_load(f)
- yamlファイル全般　-> solverとdirnameを絶対パスに変更

出力(log)
savedir, 繰り返し数, 経過時間(秒), ハードネス, 辺数, エントロピー, 最大値, 全体の最大値(リセット関係), 平均(ave = ave*(1-eps) + r*eps)

やること
- 簡単な逐次アルゴリズム(ex. クイックソート)での検証
  - https://qiita.com/take_o/items/fb303c85ce2a44afaf4d#:~:text=C++%E3%81%AB%E3%82%88%E3%82%8B (クイックソート)(一部修正)
  - NNの出力を数列に変更
    - 各インデックスにおける各数字の生成率を設定し、累積和を求める
     → 0~生成率の和の間で乱数を生成、乱数を超える最小の累積和のインデックスを数列の値として出力
    - 0~n-1までの数字に優先度を設定　←没
     → 優先度の高い順に並び変え、数列として出力
  - 出力した数列の可視化
  - ピボットの位置による差の検証(右端、左端、中央、ランダム)
  - 重複の有無による差の検証
- 分散アルゴリズムでの検証(最悪時)
   - HiSamplerでグラフを出力、出力したグラフでパスを探索
- 分散アルゴリズムでの検証(平滑時)

# HiSampler: Learning to Sample Hard Instances for Graph Algorithms

This algorithm learns generating hard instances of graph problems automatically.

https://arxiv.org/abs/1902.09700

## Getting Started

```
pip install -r requirements.txt
make solver/coloring
python3 ./hisampler.py yamls/coloring.yaml
ls results/coloring
```

## Train the Model with Your Own Solver

Your solver must take input of the following format from the standard input.

```
n m
a_1 b_1
a_2 b_2
...
a_m b_m
```

`n` is the number of nodes, `m` is the number of edges, and `(a_i, b_i)` is the endpoints of the i-th edge.
The indices of the vertices are numbered as 0, 1, 2, ..., n-1.

Your solver must output a single value which represents the hardness value (e.g., the time consumption or the number of recursive calls).

You can reuse the off-the-shelf configuration file such as `yamls/coloring.yaml` by chaning the solver path.