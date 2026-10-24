# 180epoch以降の劣化を切り分ける再学習

2026-10-23。本人の依頼は、学習方法の原因を切り分けて再学習し、未補正の出力音声で音質改善を確かめること。前回の推論用重み補正はこの依頼の完了ではない。

## 現在の状態

2026-10-24、両条件の完全checkpointを旧torchでロードし、内部epoch182、G/D全942/106 optimizer状態のstep182000、180/181/182個別モデルとtraining_complete.jsonを確認した。quarter_lrの読込障害からの復旧も成功し、対照2条件の学習部分は完了。新ホストでの復旧launcherも完了済みcheckpointを検出して両条件をスキップした。次は個別モデルを回収し、通常8文の未補正音声を原本180・従来再学習200と比較する。音質改善や劣化原因の特定はまだ確認できていない。最新のホスト・再開情報は共有Obsidianを参照する。

以下は復旧途中の記録。

2026-10-24 02:25 JST確認：controlは182完了マーカーとtraining_complete.jsonを保存済み。quarter_lrはDrive上QUESTION_025のWAV読込でsoundfile.LibsndfileErrorが発生して停止した。元GPUホストも消失していたため、新しいT4へ接続し旧環境を復元中。両条件の音質比較はまだ実施していない。

復旧用の`stage_runtime_audio`は、元ZIPをローカルディスクへコピーし、ZIPと全WAVのSHA256をpreparation.jsonと照合する。永続設定と録音は変更せず、実行用YAML/wav.scpだけローカルに作る。同一波形を読むための変更で、音質補正ではない。fixtureでbyte一致・元設定保持・破損cache修復を検査済み。新GPUでの再開batchはまだ未確認。完了controlは再学習せず保持する。進捗と復旧処理は共有Obsidianの最新再開情報を優先する。

2026-10-24 00:27 JST、通常利用のT4 GPUが回復した。既存アカウントのDriveを再接続し、旧環境を復元。Drive上に原本180から両条件を作成し、controlの開始前検査（652文の音素長、47音素順序、strict load、両optimizerのmomentum/step/LR/scheduler）を通過した。00:45頃に180checkpointから181/182epochへ再開。00:46頃に181epochの1〜50batch更新ログ（G/D LR各9.777e−5、train_time1.453秒/batch）とGPU本体PID16014を確認。両条件の品質比較は未実施で、改善完了ではない。

開始前検査のimportがroot loggerをWARNINGに設定し、ESPnetのbasicConfigだけではINFO進捗を出せない問題を実機で確認した。学習条件を変えずに設定済みlog_levelをrootへ反映する修正を加え、最初の未保存181途中のプロセスを停止。完全checkpointが180のままであることを確認して再起動し、INFOの再開・epoch開始ログが出ることを確認した。前のログはcontrol/train_before_logging_fix.logに保持。ログ修正前の実行を完了epochとして数えない。

保存先はDrive `MYCOEIROINK_WORK/controlled_retrain_20261023/pilot/`。各arm/train.log、preflight.json、model/checkpoint.pth、pilot/paired_train.logを確認する。setupログは`MYCOEIROINK_WORK/controlled_setup_20261024.log`。既存プロセスを検査してから復旧し、prepareで上書きしない。前日23:13頃のGPU割当不可は解消したが、無料枠の継続時間は保証しない。

元の180epochと再学習200の完全checkpointに保存された170〜200epochの損失を取り出した。[loss_history_170_200.json](loss_history_170_200.json)。181再開時に損失が急増した形跡はない。trainのmel損失は180で16.556、200で16.352、generator総損失は30.813→30.777。これらの値と高域周期ピークの悪化は一致しない。損失の低下を音質改善と扱えず、これだけで過学習や学習率を原因と断定しない。

## 保存済み185の追加検査

Driveに保持されていた前回再学習の`checkpoint_185.pth`を回収し、内部epoch185、両optimizer942/106件の全step185000、scheduler185、モデル1049テンソルの有限値を確認した。SHA256 `b539824e093a902e658df3414dbd7447a2ea6fb44548d59688da9804b1a7b33b`。重みを補正せず抽出し、同じ隔離エンジンで通常8文を生成した。アプリ登録は変更していない。

[intermediate185_comparison.json](intermediate185_comparison.json) の高域周期ピーク指標平均は180=0.265、185=0.213、200=0.473dB。185は180に対して4文で上昇・4文で低下。185→200では7文で上昇したが、感謝文は185の1.130→200の0.795と逆に低下した。全文章で同じように単調悪化したわけではない。一方、185の母音区間8〜20kHz/300〜4000Hz比は全8文で180より高い。周期ピークの平均だけで185を高音質と選ぶこともできない。ここから特定の設定を劣化原因とは断定できない。

生成音声は全8文44.1kHz/16bit/mono・PCMクリップ0。本人文の比較`controlled_retrain_20261023/intermediate185/listening/user_180_185_200.wav`は180→185→200の順、1秒の間隔。線形ゲインだけで各音声−23LUFSへ揃えた。EQ・時間伸縮・ノイズ除去・重み補正なし。これは前回学習の中間地点の検査であり、今回の対照再学習が始まったという意味ではない。

再現：`tools/intermediate_retrain_compare.py prepare`で完全checkpointの内部状態を検査して隔離エンジンへ登録し、`generate`で通常8文を生成・測定する。元のモデルや本番アプリには書き込まない。

## 最初の対照実験

| 条件 | control | quarter_lr |
|---|---:|---:|
| 出発点 | 無加工180完全checkpoint | 同左 |
| 初期G/D学習率 | 各9.777498621e-5 | 各2.444374655e-5 |
| 学習範囲 | 181〜182 | 181〜182 |
| 1epochの更新数 | 1000 | 1000 |
| モデル・録音・音素順序・seed・batch・損失 | 共通 | 共通 |

G/D双方の学習率を同率で変えるため、まず全体の更新幅の影響を調べる実験。GとDそれぞれの寄与を分離した実験ではない。0.25倍は検証する候補であり、改善が確認された設定ではない。

- 未補正180完全checkpointのSHA256 `ab6925215af0ea111939313749c64bab66743eef90411d9f30165c901687b097` を固定。
- 両条件のモデル1049テンソルは原本と全件一致。optimizer942/106件のstep・momentum等も全件一致。変えたのは各param groupのlr/initial_lrとschedulerのbase_lrs/_last_lrのみ。schedulerのepoch/gammaは維持する。[integrity_checks.json](integrity_checks.json)。モデル、momentum、schedulerの意図しない変更を入れる負例も検査で拒否した。
- 録音652WAVは元ZIPのバイト列を保存し、44.1kHz/16bit/mono、全IDと歴史的な波形長に一致。元ZIP hash `e34f05d23bf01b124658e8f2f1f6257d6834a053f5293c4754122b968c452915`。
- ESPnet202211、torch1.12.1+cu113、NumPy1.23.5を要求。開始直前にCUDA、strict model load、両optimizerのmomentum形状・有限値、全652文の音素長を検査する。10月24日にcontrolの実行前検査をT4で通過。quarter_lrは順番が来た時点で同じ検査を行う。
- 設定の差は出力先、optim_conf、optim2_confのみ。seed777、numel batch、元の損失と学習データを維持。cuDNN deterministic=falseも元のままであり、GPU計算のビット単位の再現性は保証しない。
- Mac側の準備はPython3.10、torch1.12.1（CPU）、NumPy1.23.5でも実行し、両条件の保存ファイルを再ロードして上記一致を確認した。最初に最新NumPyで保存した検査用コピーは旧NumPyにない`numpy._core`を参照して読み込めなかったため、学習用と混同しない別ディレクトリへ隔離した。Colabでは必ず旧環境で原本からprepareする。Mac上のCUDA実学習を確認したという意味ではない。
- validは元と同じ先頭5本でtrainに重複する。原因切り分けの対照を揃えるため初回は分割を変えない。未知音声の検証データとは扱わない。
- 180/181/182の個別モデルを保持。過去の未取得モデルを参照する終了時平均化は無効にし、学習回数で付くbestという名前を音質順位に使わない。学習本体の順伝播・逆伝播・損失は変更しない。

## 起動と再開

ソース `tools/controlled_retrain.py` と `tools/controlled_start.sh`。従来RUNの200や推論補正済みモデルから開始しない。元のRUNを上書きしない。

無料GPUとDriveを接続後、既存の `tools/retrain_setup.sh` で旧環境を準備する。以下はColabのターミナルで実行する。`repo` はこのソースリポジトリをcloneしたディレクトリ。

```bash
source /content/tokine-202211/bin/activate
work=/content/drive/MyDrive/MYCOEIROINK_WORK
pilot="$work/controlled_retrain_20261023/pilot"
repo=/content/hanazono-tokine-coeiroink
audio_zip="$(python - "$work" <<'PY'
from pathlib import Path
import sys
paths = list(Path(sys.argv[1]).glob('*/花園トキネtalk_full_wavs.zip'))
assert len(paths) == 1, paths
print(paths[0])
PY
)"
python "$repo/tools/controlled_retrain.py" prepare \
  --reference "$work/retrain_180_20261021" \
  --destination "$pilot" \
  --audio-zip "$audio_zip" \
  --corpus "$work/marged-corpus.txt"
nohup bash "$repo/tools/controlled_start.sh" "$pilot" > "$pilot/paired_train.log" 2>&1 &
```

prepareは既存の実験先を上書きしない。一度準備した後の切断復旧ではprepareを再実行せず、保存したpilotを引数にcontrolled_start.shを起動する。データをDrive側へ置くのでセッション消失後もパスが維持される。launcherの同時起動はホスト内のflockで拒否する。起動前には別の学習プロセスが動いていないことも確認する。

ローカルで準備済みのpilotは `output/talk_full_20260917/controlled_retrain_20261023/pilot/`。Macの絶対パスを含むため、この設定をColabへそのままコピーして使わない。

## 改善の判定と次の分岐

回収後の `tools/controlled_pilot_compare.py generate --root <比較先>` が48WAVとresults.jsonを保存したら、`python tools/controlled_pilot_report.py <比較先>` でCSV、数値集計、通常8文の同音量比較を生成する。各文の並びは180原本→従来再学習200→control181→control182→quarter_lr181→quarter_lr182。181/182別の180→control→quarter比較も保存する。線形ゲインと1秒間隔だけを加え、波形の長さを維持する。通常は−23LUFS、ピーク余裕が足りない文は全条件共通の低い目標値に下げる。summaryのquality_improvedは未判定、listening_verifiedはfalseのままとし、数値出力で聴感検証を代用しない。既存180音声8文を使った隔離fixtureで48ゲイン書出し・24連結の動作、サンプル数、音量、ピークを確認済み。候補モデルの音質検証は回収後に行う。

1. 各条件の181/182から、本人指定文を含む通常文8本を同じCOEIROINKで生成する。未補正180・未補正再学習200と比較する。母音伸縮、EQ、ノイズ除去、推論用重み補正を加えない。
2. 約689Hz間隔の高域ピークだけでなく、音量、高域全体の量、発音、クリップ、声質への影響を併せて比較する。聴き比べ用のみ線形ゲインで同音量に揃える。数値だけで聴感確認済みとはしない。
3. 学習率変更で改善しなければ、その設定を採用せず、損失の周波数分解能やGANのG/D更新の寄与を次の対照実験で調べる。結果なしに変更を重ねない。
4. 候補の改善を確認してから追加学習する。200まで到達したことだけで完了とせず、未補正音声での改善確認まで継続する。アプリや公開ZIPは検証前に置換しない。

現時点で劣化原因の特定、修正学習の完了、音質改善は未達。
