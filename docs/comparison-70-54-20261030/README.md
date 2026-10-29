# 70・54epochの本人指定台詞比較

「すごい！ 全部言葉になったよ！」を、旧原本70/54epochで速度1生成。既存noise_20261021のモデルSHAを照合し、隔離engine50033を使用。現COEIROINKの131/180登録・配布版・作業中の文章/調声は変更していない。

OUTPUT/compare_70_54_kotoba_20261030にraw/matched/request/本文/manifest/compare_70_54.wavを保持。70→54・1秒間隔。音量合わせは一定ゲインのみ、EQ/ノイズ除去/母音伸長/推論重み補正なし。44.1kHz16bitmono、clip0、元サンプル数保持、一定ゲイン誤差0.6LSB未満、連結PCM一致・LUFS/true peakを検証。実聴は未確認。

再現：tools/top_three_long_compare.py --root OUTPUT/compare_70_54_kotoba_20261030 --epochs 70 54 --audit OUTPUT/noise_20261021 --text-file OUTPUT/compare_70_54_kotoba_20261030/input.txt
