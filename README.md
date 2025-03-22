# Auto-Generating-Inputs-For-Smoothed-Analysis-In-Message-Passing-Model

HiSampler
- https://github.com/joisino/HiSampler

python 3.8.0
- https://qiita.com/chem-it-village/items/7c55881bcd4ca33e6327#python%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB
- https://qiita.com/syusuke9999/items/9d35bcdb4119a1c4957a

C++の環境構築
- https://www.javadrive.jp/cstart/install/index6.html
- https://www.binarydevelop.com/article/libgcc_s_dw21dll-36074

pip install(chainer)
- numpy 1.23.0
- chainer 7.8.1
- matplotlib 3.7.5
- pyyaml 6.0.2

pip install(tensorflow)
 - numpy 1.19.5
 - tensorflow 2.4.0
 - matplotlib 3.4.0

make
- https://qiita.com/ryuso/items/cf0c1d83544103cacf05

gpuの設定
- https://www.kkaneko.jp/tools/win/chainer.html
- yamlファイルのgpuを0に変更

yamlファイル
 - ハイパーパラメータの指定用(頂点数など)
 

オリジナルのHiSampler → graph.py
 - 実行方法 -> オリジナルのHiSamplerを参照(https://github.com/joisino/HiSampler)
 - オリジナルのHiSamplerからの変更点
    - util.py 63line　-> conf = yaml.safe_load(f)
    - yamlファイル全般　-> solverとdirnameを絶対パスに変更 -> 相対パスでもOK(原因不明)
 - HiSamplerの出力(log)
    savedir, 繰り返し数, 経過時間(秒), ハードネス, 辺数, エントロピー, 最大値, 全体の最大値(リセット関係), 平均(ave = ave*(1-eps) + r*eps)
 - yaml -> clique, coloring, vartex_cover_approx, vartex_cover

クイックソート用のプログラム → sequence.py
 - 実行方法 -> graph.pyと同様
 - ピボット位置の変更 -> yamlファイルで指定
    - c...center
    - l...left
    - r...right
    - rnd...random
 - 数列の重複ありなしを指定可能(gen_sequence関数(line38))
 - ランダムに生成した数列での実験 -> random_sequence_generation.py

メッセージパッシングモデル(単一始点最短経路問題)
  → MNN_tensorflow.py ... DQNのみのプログラム(AlphaZeroの教科書をベースに作成)
  → MNN_tensorflow_rainbow.py ... DQN + rainbow(参考 https://horomary.hatenablog.com/entry/2021/02/11/173638)
  → MNN_tensorflow_rainbow_smoothed.py ... 平滑モデル導入ver
  → MNN_tensorflow_rainbow_plus_BBF.py ... rainbow + BBF(参考 https://horomary.hatenablog.com/entry/2024/11/03/140544) (学習がうまくいかないため、何らかの不具合がある可能性あり)
 - tensorflowを使用
 - yaml -> shortest_path
 - ランダムによる実験 → shortest_path_random_select.py
 - 最悪時(一番大きいメッセージを送信) → shortest_path_worstcase_message.py