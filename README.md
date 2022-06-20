# OpenCV_Rock_Paper_Sciccors
OpenCVの共通点マッチングを用いたじゃんけんの手の判別

概要
最初にグー、チョキ、パーの順番に手の形を登録する。(肌色認識で枠内の肌色のみを検出する)
次に、枠の下にどの手に対して勝ち負けあいこが指定されるため、指定された手を枠内に出す。
別ウィンドウに答えの画像(最初に登録したもの)とじゃんけんで出したもの(各回で出したもの）の特徴点同士をつなぎ合わせた画像が表示される。
合否は分からないが、特徴点を出すことで同じような手の形であることがわかる。

実行方法
.slnファイルとフォルダを同じディレクトリに置く。
Visual Studio にて.slnファイルを指定して開く。
デバッグする。
