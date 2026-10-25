# 100・180epoch長文試聴（2026-10-26）

本人指定100/150/180のうち、保存済み旧原本100/180を前回と同じ全文・速度1で生成。150は独立棚卸しnoise_20261021/inventory.jsonのmissing（148〜159）に含まれ、モデルフォルダと準備フォルダのファイル検索でも見つからない。別epochの代用や補間はしていない。

```sh
python3 tools/top_three_long_compare.py --root /Users/aaa/Documents/COEIROINK/花園トキネ_MYCOEIROINK準備/output/talk_full_20260917/long_comparison_100_180_20261026 --epochs 100 180 --audit /Users/aaa/Documents/COEIROINK/花園トキネ_MYCOEIROINK準備/output/talk_full_20260917/noise_20261021
```

モデルSHA照合、44.1kHz16bitmono/クリップ0/長さ保持/一定ゲイン誤差0.6LSB未満/連結PCM一致を実検証。rawは無加工、matchedは一定ゲインのみ。compare_100_180.wavの順序は100→180、1秒間隔。長さ・音量・出力SHAはmanifestに記録。2モデル時の合計時間の記載を実WAV長に合わせ、スクリプトもモデル数に応じた間隔計算へ修正（音声自体の連結と検証は正常）。

EQ・ノイズ除去・母音伸長・推論重み補正なし。アプリ登録・配布物・学習停止は維持。生成用50033停止と本番50032生存を確認。実聴や発音確認は未実施。
