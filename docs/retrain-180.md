# 180epochからの比較再学習

2026-10-21。181再開時の実装・音素表・optimizer不整合を避け、元の180epoch終了時点から181〜185を比較する。改善を検証する前に200へ延長しない。

## 復元した学習状態

Google Driveのcheckpoint.pthの第81版（2026-10-05 21:00 JST）を別ファイルへ取得した。元ファイルの現行版は変更していない。

- SHA-256: `ab6925215af0ea111939313749c64bab66743eef90411d9f30165c901687b097`
- 1,119,369,272 bytes、reporter.epoch=180。
- 1049テンソルすべてがローカル180epoch原本と完全一致。
- generator optimizer: 942パラメータ、discriminator optimizer: 106パラメータ。momentumを含む状態が残る。
- 2つのschedulerはlast_epoch=180、学習率はともに`9.777498621170277e-05`。

重みだけからの追加学習ではなく、完全checkpointからのresumeを使う。復元失敗を無視する変更は加えない。ノイズ補正や音量校正をした推論用派生モデルは学習に使わない。

## 実行構成

専用Python 3.10 venv、ESPnet 202211、PyTorch 1.12.1+cu113、NumPy 1.23.5。元ノートブックのshirowanisan/espnet 0.10.3（commit `abab1e48c06eddb4bd3bc8762176317d12d2b192`）のrequirements-cuda113.txtを使用する。元config.yamlのモデル構造、47項目のtoken_list、seed、batch、optimizer、schedulerの設定を維持する。180時点のPyTorch/CUDAを含む全パッケージの完全一致までは確認できていない。

元の652本ZIPを無加工で使う。初案の公式トリム・peak normalizeは過去のspeech_shapeと一致せず、学習前に停止した。10月5日1:31の第18版speech_shapeとfeats_stats.npzを回収し照合すると、元ZIPのSTFTフレーム数207600が完全一致し、無加工波形のスペクトルsumの相対差は約1.06e-5、sum_squareは約5.88e-7、推定ゲイン比は0.99999958だった。したがって、操作履歴にある正規化コードの記載ではなく、保存された実測統計に合わせて元ZIPを使う。トリム不一致を181以前と以後の変更の証拠には扱わない。

過去のspeech_shapeとtext_shape.phnを使用し、全ID・波形長に一致しなければ停止する。公式local/data.shはtrainを全652本、validを先頭5本にしており、この既存の分割を引き継ぐ。trainとvalidが重なるため、valid lossを未知文への汎化性能の証拠には使わない。既存ディレクトリは変更しない。

Driveの`MYCOEIROINK_WORK/retrain_180_20261021/`に保存する。モデル出力はその下の`model/`、上限は185epoch。既存の`exp/tts_mycoe_model`は出力先にしない。

実行用ソース:

- `tools/retrain_setup.sh`: 旧環境の準備。
- `tools/retrain_prepare.py`: 入力検証、前処理、専用設定の作成。
- `tools/retrain_preflight.py`: strictなモデル復元、両optimizerの状態・momentumの形状と有限値を検査。
- `tools/retrain_start.sh`: 検査成功後に学習開始。

## 200epochへの続行（2026-10-22）

本人からの明示指示『200epochまでやらせて』により、旧185上限／200延長禁止を撤回し、185完全checkpointから186〜200epochへ学習を続行する。
- 185checkpoint（`model/checkpoint.pth`）および `train.yaml` の別名バックアップ（`checkpoint_185.pth`、`train_185.yaml`）を作成して整合性確認後にresume。
- 旧exp200や補正済みモデルは使用禁止。
- 47音素順序、無加工652WAV、ESPnet202211、torch1.12.1+cu113、NumPy1.23.5、optimizer/scheduler設定を完全維持し、`max_epoch: 200` のみ更新。
- `retrain_prepare.py` と `retrain_preflight.py` の185固定ガードを200へ更新。


まず181〜185の同文音声を生成し、180原本に対する約689Hz間隔の周期ピーク、音量、発音を比較する。EQやノイズ除去で比較結果を加工しない。高域改善が未検証のまま既存COEIROINK登録や配布ZIPを置き換えない。

中断時は実プロセス・GPUとログの時刻を確認し、重複起動を避ける。新しい専用checkpointがあれば、そのepochと整合性を検証してresumeする。再開用ソース・データの配置は同じに保つ。GPU上限の回避や課金変更は行わない。

## 起動確認（2026-10-21）

20:22:05 JST、T4の既存セッションで181/185epochを開始。20:24:32に181epochの51〜100batch、GPU使用率96%・8467MiBを確認した。事前検査はモデルstrict load、両optimizerのmomentum形状・有限値、652文の音素長に合格。改善の判定は学習後の比較まで未確定。

監視は本人の希望によりAntigravityのGemini 3.8 Flash Mediumへ引き継ぐ。15分ごとの進行・保存・停止確認と既存手順での復旧を担当し、品質比較はCodexが担当する。稼働中の再起動、学習設定の独断変更、185以降への延長は行わない。引き継ぎ先の定期実行を確認してからCodex側の重複監視を停止する。
