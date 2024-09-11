# Auto-Generating-Inputs-For-Smoothed-Analysis-In-Message-Passing-Model

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

util.py 63line
 -> conf = yaml.safe_load(f)

 yaml
  -> solverとdirnameを絶対パスに変更

gpuの設定
- https://www.kkaneko.jp/tools/win/chainer.html
- yamlのgpuを0に変更
