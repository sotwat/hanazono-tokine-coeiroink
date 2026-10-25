# 145・141・131epoch 長文試聴（2026-10-26）

本人が選んだ旧原本145/141/131epochで、111/117/134の比較と同一全文・速度1を使用。既存の隔離engine50033を使い、登録アプリ50032・公開配布物は変更していない。3モデルSHAは既存独立棚卸しと照合、configはいずれもc56c70c8b6fea3fdd679eee3dea23be622a6f5a762ffa2b33c660b5795c5509b。

```sh
python3 tools/top_three_long_compare.py --root /Users/aaa/Documents/COEIROINK/花園トキネ_MYCOEIROINK準備/output/talk_full_20260917/long_comparison_145_141_131_20261026 --epochs 145 141 131 --audit /Users/aaa/Documents/COEIROINK/花園トキネ_MYCOEIROINK準備/output/talk_full_20260917/noise_20261021
```

各raw.wavは無加工生成音。matched.wavは前回と同じ一定ゲインだけの音量合わせ。compare_145_141_131.wavは145→141→131、発話間は各1秒、全長151.03秒。44.1kHz/16bit/mono、クリップなし、全発話長保持、一定ゲイン誤差0.6LSB未満と連結PCM一致を実検証。本文・モデル/出力SHA・音量・長さはmanifest参照。loudnormフィルタの出力は破棄して測定にのみ利用し、input_*が保存音声の測定値。加工はvolumeフィルタのみ。

学習・ASR・旧比較生成は実行していない。EQ/ノイズ除去/母音伸長/推論重み補正なし。終了後50033の停止、50032の生存を確認。実聴や聴感上の順位・発音確認は未実施。134の本人ノイズ評価と低ノイズ推奨撤回は維持する。
