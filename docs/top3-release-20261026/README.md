# 111・117・134epochの登録と配布（2026-10-26）

本人の指定により、既存133候補の固定周波数・周期性成分の代理指標上位3件をMacへ登録し、配布版v1.2.0に収録する。追加の指定により、Macの花園トキネはこの3件だけを登録する。他のキャラクターは対象外。

元の学習方式の未補正モデルを使用。モデルSHA・設定SHA・選定指標・閾値感度・ZIP情報は[package.json](package.json)。134epochの順位範囲は3〜95位で、聴感や総合音質の上位3件を保証しない。実聴確認は未実施。学習原因の特定と未補正モデルの音質改善は引き続き未達。

## 検証

- パッケージ内の3モデルを元モデルSHAへ照合。サンプル9本は同じ通常8文評価の既存cacheからSHA一致で再利用。重み・音量補正、EQ、ノイズ除去なし。
- 全ZIPエントリのCRCと展開内容SHAを検査。READMEの導入方法は実施したspeaker_info直接配置に統一。
- 実アプリの50032エンジンで各モデルの通常本人文を速度1で生成。3本とも評価cacheのPCMと完全一致。[installed_validation.json](installed_validation.json)。既存8文測定の再生成はしていない。
- 再起動後の実APIで花園トキネのstyleIdが1111・1117・1134だけであることを確認。[registration.json](registration.json)。実アプリのキャラクター設定画面でも3行のみを確認。
- 開いていた作業は「上位3登録前_20261026.cink」へ保存。複製「上位3登録後_111epoch_20261026.cink」はスタイル名とIDだけを111へ変更し、その他のテキスト・調声・F0を保持。最終的に複製を実画面へ読み込み、111epoch・話速1.07・高さ0.05・RESAMPLINGを確認。

## ローカル復元

対象は `/Applications/COEIROINKv2.app/Contents/MacOS/speaker_info/hanazono-tokine-2.1.0`。既存のフォルダ名は維持し、metasのversionを2.2.0、登録stylesを3件にした。旧20スタイルのモデル・設定・画像は削除せず保持した。

バックアップは準備フォルダの `output/talk_full_20260917/top3_release_20261026/metas.before.json`。復元時はCOEIROINKを終了してこれを対象のmetas.jsonへ戻し、再起動する。旧プロジェクトを使う場合も先に旧登録を復元する。

再梱包は `tools/package_top_three.py` の3引数（`--output` 学習output、`--destination` 新規配布フォルダ、`--installed` 画像・規約の既存話者フォルダ）で行う。既存配布物の上書きを拒否する。
