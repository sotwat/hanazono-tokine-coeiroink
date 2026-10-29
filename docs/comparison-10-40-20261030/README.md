# 10〜40epoch保存モデルの本人指定台詞比較

本人は直前の70/54試聴で70を採用と明言。次に10〜40の比較を依頼。同範囲の保存モデルは10/15/20/30/40の5件なので、初期モデルの発音品質を固定周期指標だけで除外せず全5件を比較。台詞は直前と同じ「すごい！ 全部言葉になったよ！」、速度1。

既存隔離engine50033と無補正原本、SHA照合済み。OUTPUT/compare_10_40_kotoba_20261030にraw/matched/request/本文/manifest、10→15→20→30→40・各間1秒の連結音声を保存。一定ゲインのみ、EQ/ノイズ除去/母音伸長/推論重み補正なし。44.1kHz16bitmono/clip0/元サンプル数保持/ゲイン誤差0.6LSB未満/連結PCM一致/LUFS/true peakを検証。実聴・発音の聴覚確認は未実施で、最優秀の断定なし。

登録と配布版は比較中のため変更なし。70採用は本人の実聴選択として記録し、物理登録済みとは扱わない。学習・監視は停止維持。再現はtools/top_three_long_compare.py --epochs 10 15 20 30 40 --audit OUTPUT/noise_20261021 --text-file OUTPUT/compare_10_40_kotoba_20261030/input.txt --root OUTPUT/compare_10_40_kotoba_20261030。
