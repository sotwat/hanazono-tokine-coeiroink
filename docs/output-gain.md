# 200epochの音量補正（2026-10-20）

v1.1.0の200epochは、同じ文章・音量1.00で生成した100epochより平均11.26dB小さくなっていました。画面のプリセットを介さないAPIでも再現したため、モデルの出力レベルとして対処しています。120〜190epochでも同程度の低下があり、101epochの最終出力係数が100epochの約半分になっていることを確認しました。学習時の変化を引き起こした具体的な操作・環境は未確定です。

## 変更範囲

200epoch原本の最終畳み込み層にある次の2テンソルだけに、測定した係数3.799を掛けました。その他1,047テンソルはtorch.equalで原本との完全一致を確認しています。

- `tts.generator.decoder.output_conv.1.weight_g`
- `tts.generator.decoder.output_conv.1.bias`

これは最終Tanhの前のゲイン補正です。厳密な線形のWAV増幅とは異なり、強いピークはTanhで緩やかに抑えられます。100epochのモデルは変更していません。再学習や単語・音素の対応表の変更は行っていません。

## 測定

COEIROINK 2.13.0・CPU・44.1kHz/16bit/mono、音量/話速/抑揚1.00、ピッチ0、前後無音0.1秒。FFmpegのloudnormでintegrated LUFSとtrue peakを測定しました。

比較文：

1. こんにちは、花園トキネです。これからもよろしくお願いします。
2. 今日はいい天気ですね。一緒にお出かけしませんか。
3. 東京から京都まで、新幹線で旅行します。切符を忘れないでください。

3文の算術平均は100epochが−18.44 LUFS、200epochの補正前が−29.70 LUFS、補正後が−18.45 LUFSです。200epochは追加4文を含む7文でtrue peak最大−1.53dBTP、PCMクリップ0を確認しました。文章や生成条件により音量差は残ります。任意の文章や音量設定でのクリップを保証するものではありません。

ローカルの中間スタイル110/120/130/140/160/170/180/190epochもそれぞれ別の係数で校正し、同じ3文の平均を100epochの0.1dB以内に合わせました。150epochは原本がないため対象外です。公開ZIPに収録するスタイルは100epochと200epochの2つです。

3文のWhisper-smallによる音声認識結果は補正前と同様で、固有名詞・「切符」の認識差は残っています。音量補正で読み上げ品質まで改善したとは判断していません。比較2文は、原波形から計算したTanhゲイン変換と実際の出力の相関が0.9999999以上でした。もう1文は再生成で512サンプルの長さ差があり、この波形一致判定の対象外です。

集計値は[検証結果JSON](output-gain-verification.json)を参照してください。

## 再現

原本200epochのSHA-256：`e2be393a46ffbba9153dda15b73b7fca63fc474ff504a93219a45c34a7c21d9e`

```sh
uv run --with torch python tools/calibrate_output_gain.py 200epoch-original.pth 200epoch-calibrated.pth --gain 3.799 --expected-sha256 e2be393a46ffbba9153dda15b73b7fca63fc474ff504a93219a45c34a7c21d9e
```

原本の上書きと既存出力への上書きを拒否します。補正済みモデルへ二重適用しないでください。出力モデルと同名のJSONに、原本/出力ハッシュ・係数・変更テンソルを記録します。出力モデルを同じスタイルのpthと置き換え、COEIROINKを再起動して検証してください。

ESPnetの[HiFiGANGenerator実装](https://github.com/espnet/espnet/blob/v.202211/espnet2/gan_tts/hifigan/hifigan.py)で最終Conv1d→Tanhとweight normalizationを確認してから適用しています。
