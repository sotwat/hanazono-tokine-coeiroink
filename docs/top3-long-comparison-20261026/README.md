# 111・117・134epochの長文比較（2026-10-26）

本人の聴き比べ依頼で、同じセリフをMacに登録済みの原本111・117・134epochに速度1で生成した。各モデルSHAを登録先で照合。全文はmanifest.jsonと再現スクリプトのTEXTに保存。

```sh
python3 tools/top_three_long_compare.py --root /Users/aaa/Documents/COEIROINK/花園トキネ_MYCOEIROINK準備/output/talk_full_20260917/top3_long_comparison_20261026
```

出力先の各`*_raw.wav`は無加工原音、`*_matched.wav`は一定ゲインだけで音量を揃えた版。`compare_111_117_134.wav`は111→117→134の順、間隔は1秒。発話はそれぞれ49.60/49.75/49.08秒、比較全体は150.42秒。全文長保持、44.1kHz/16bit/mono、クリップなし、一定ゲインとの差は0.6LSB未満、連結PCMは各出力と完全一致を実検証。LUFS・true peak・SHAはmanifestに記録。

FFmpeg loudnormは測定出力を捨てる解析にだけ使用。加工はvolumeフィルタのみで、EQ・ノイズ除去・コンプレッサ・母音伸長・推論重み補正は不使用。raw_loudness/matched_loudnessのinput_*が保存WAVの測定値であり、output_*とnormalization_typeは破棄した解析出力に関する値。

実聴・発音内容の聴覚確認は未実施。音質順位の追加判定はしない。追加学習と自動監視の停止を維持し、既存COEIROINKの作業プロジェクト・配布版v1.2.0は変更していない。
