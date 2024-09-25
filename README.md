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

出力
savedir, 繰り返し数, 経過時間(秒), ハードネス, 辺数, エントロピー, ma, global_ma, ave

やること
- 簡単な逐次アルゴリズム(ex. クイックソート)での検証
  - https://qiita.com/take_o/items/fb303c85ce2a44afaf4d#:~:text=C++%E3%81%AB%E3%82%88%E3%82%8B (クイックソート)(一部修正)
  - NNの出力を数列に変更
   - 各インデックスにおける各数字の生成率を設定し、累積和を求める
     → 0~生成率の和の間で乱数を生成、乱数を超える最小の累積和のインデックスを数列の値として出力
   - 0~n-1までの数字に優先度を設定
     → 優先度の高い順に並び変え、数列として出力
  - 出力した数列の可視化
- 分散アルゴリズムでの検証(最悪時)
- 分散アルゴリズムでの検証(平滑時)
