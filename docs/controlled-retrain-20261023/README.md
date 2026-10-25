# 180epoch以降の劣化を切り分ける再学習

## Ranking aggregation remains provisional (2026-10-25)

The current combined inventory has 135 file entries / 134 distinct file hashes, and 133 distinct tensor states after canonicalizing the two epoch-180 serialization aliases. Previous proper-retrain epoch189 has been extracted and audited; the older extraction-pending statements below are historical. Matched evidence currently covers 99 models / 792 ordinary utterances. Their loudness and local ASR collection is complete; all WAV hashes and finite loudness values were verified, zero clips and no RMS drop greater than 6 dB from the same-text original180 were found. ASR is a content screen, not listening or pronunciation verification. See `ranking_validity_validation.json`.

`tools/ranking_aggregate.py --root <controlled_retrain_20261023>` reuses the index, complete spectral peak lists and validity summaries. It applies the frozen lexicographic rule and all 27 threshold combinations, and writes `ranking_evidence/aggregate_provisional.{json,csv}` with per-text evidence, spread, loudness and sensitivity ranks. Missing control/quarter epochs183–199 (34 models), content review and broad-harshness/listening verification prevent a confirmed global top five. The output deliberately leaves `global_top_five_confirmed=false`, even if the numerical collection later becomes complete. Do not infer a training cause or overall quality from the proxy order.

Add `--make-listening` to export the original180 baseline followed by every candidate with provisional rank <=5 (including ties), using only constant gain and one-second gaps. `listening_provisional/manifest.json` records source/output hashes, unchanged utterance lengths, LUFS and true peaks. The current 48 normalized ordinary utterances and eight joined files are verified; no actual listening was performed. Reviewed ASR text differences remain unseparated from possible pronunciation errors. See `ranking_aggregate_validation.json` for the limited evidence and sensitivity ranges.

## Ranking collection: original-series measurement complete (2026-10-25)

Both training arms are complete; do not restart training. The ranking rule is frozen in [ranking-protocol.md](ranking-protocol.md). [ranking_inventory.json](ranking_inventory.json) currently covers 81 saved original models, 42 control/quarter files and 11 previous-retrain models, 134 file entries / 133 distinct file SHAs. The shared control/quarter epoch-180 SHA is an alias. The previous proper-retrain individual-model inventory covers 191–200 (10 models); epoch185 is already extracted and evaluated locally; epoch189 backup extraction is still pending; this is not a complete candidate list or a top-five result.

`tools/ranking_original_survey.py` completed all eight canonical ordinary sentences for the 81 original models: 648 waveforms. All WAV hashes, 44.1 kHz/16-bit/mono format, zero PCM clips and finite primary metrics were verified. See ranking_original_validation.json. The isolated engine exited and port 50033 is free; production 50032 remains unchanged. Output is `OUTPUT/controlled_retrain_20261023/ranking_original/{results.json,raw,engine.log,provenance.json}`. The existing 162 user/travel waveforms were checked against recorded model and WAV hashes and reused. Original and retrain comparison engine binaries have identical SHA256 `87dc472aa450fd6510111e0e0f3c9fccc55d141d20e560202b10a15e6adc79db`. complete.json contains 81 models / 648 waveforms. Reuse these outputs; do not regenerate the complete original survey. Do not rank partial coverage as global top five.

Reproduction/resume command (only for missing evidence; do not rerun the completed collection):

```sh
python tools/ranking_original_survey.py --root "$OUTPUT/controlled_retrain_20261023/ranking_original" --previous "$OUTPUT/noise_20261021" --cache "$OUTPUT/noise_onset_20261024"
```

Use the existing analysis Python environment. Source caches, model files and requests are verified before reuse. Interrupted WAV/JSON pairs are rejected for inspection rather than overwritten. The remote inventory completed previous proper retrain model/191epoch.pth through 200epoch.pth (10 SHA-checked models) and was recovered via the existing Drive UI to ranking_inventory_previous.json; preserve both checkpoint backups; epoch185 is already extracted locally and epoch189 still needs verified extraction. Colab connection was stale after inventory completion; notebook reload showed disconnected T4. No GPU reconnection was requested solely for inventory recovery. Remote log `/content/previous_rank_inventory.log`, result Drive `MYCOEIROINK_WORK/retrain_180_20261021/ranking_inventory.json`. No training was launched. All global rankings, broad-harshness assessment and listening remain pending.

## Quarter LR epoch 200 comparison (2026-10-25)

Quarter LR completed epoch 200. After the old host ended, a normal T4 connection on host 85c5778453ef and the existing Google Drive mount recovered the actual files. Complete checkpoint epoch and both schedulers are 200; G/D optimizer states contain 942/106 entries, all at step 200000 with finite momentum. The individual model has 1049 finite tensors, exactly matching checkpoint model tensors. SHA256: `afb53f814ce406dc39f2965bc2783cd187fe05549d92fced972fa2bd0f107c70`. No duplicate training was started.

[Measurements, ASR and audit](quarter200_comparison.json). The same eight ordinary sentences give periodic-peak means of 0.265 dB for original 180, 0.473 for previous retrained 200, and 0.318 for quarter LR 200. Four sentences decreased and four increased relative to 180. This does not establish consistent noise or overall quality improvement. Listening verification remains unperformed. High-band energy is not noise quantity.

All 24 raw files have zero clipped samples. Listening files use linear gain only, -23.02 to -22.99 LUFS, maximum -4.22 dBTP, with original sample counts preserved. No vowel stretching, EQ, denoising or inference weight correction. The user sentence's ASR preserves content; travel/library recognition differences remain unseparated from possible pronunciation errors.

Local results: `controlled_retrain_20261023/quarter200_comparison/`. Listening order for `user_180_candidate200.wav` is original 180 then quarter 200. The three-way file adds previous retrained 200 in the middle. Transfer ZIP: 373255339 bytes, SHA256 `36b4def58253f31982a4117a095f6ec7f3fc13c84ca13b09cdec1e99dda9b6ec`; all six parts and combined ZIP verified. Temporary Drive-only download/popup permissions were restored to defaults.

Both arms reaching 200 does not complete the quality-improvement request. The requested top five across saved 100-200 models is still pending uniform eight-sentence evaluation. Initial skill research found no dedicated ranking skill; adjacent `kunitoki/sonic-skills/audio-artifacts-debug` addresses realtime DSP faults, insufficient for TTS ranking, so it was not installed. Primary-source MoSQITo suitability/license research has started but no adoption decision or new metric implementation has occurred.


2026-10-23。本人の依頼は、学習方法の原因を切り分けて再学習し、未補正の出力音声で音質改善を確かめること。前回の推論用重み補正はこの依頼の完了ではない。

## control200 completion record (2026-10-24)

2026-10-24 19:46 JST、controlの完全checkpoint内部epoch200、両scheduler200、両optimizer942/106件の全step200000とmomentum有限値を確認した。未補正200epoch.pthの1049テンソルは完全checkpointのmodelと全件一致。SHA256 `bab9630ac4e7a9da85d1097c4e8adfb7e7b42b2ed91952b0ef92abf193bf5730`。quarter_lrは同じlauncherから183epochへ進み、GPU PID244617の301〜350batch、LR2.444e−5を確認した。ブラウザのアイドル切断表示から同じホストe5080efa989bへ再接続し、学習本体の継続を確認。重複起動なし。

control200の回収・通常8文比較は完了した。[数値・ASR・checkpoint監査](control200_comparison.json)。周期ピーク指標平均は180=0.265、従来再学習200=0.473、今回control200=0.576dB。今回control200は180より6/8文で上昇・2/8文で低下した。母音の高域成分比平均は−28.637→−28.695dBと微減するが、これをノイズ改善とは扱わない。3dB超のグリッド成分は180で公園文に2本、今回control200で本人文・図書館文・会議文・感謝文に各1本。全条件で「6dB超の突出が母音フレームの25%超に存在するピーク」の増加は確認せず、周期指標の差をそのまま知覚ノイズ量・SNRに変換しない。**180よりノイズや総合音質が改善したとは確認できない。聴感確認は未実施。** quarter_lrは200まで継続し、結果を見ずに別条件へ切り替えない。

24WAVは44.1kHz/16bit/mono、PCMクリップ0。180と従来再学習200の16本は前回と完全一致するキャッシュを使い、requestの通常8文・style・速度1を照合した。今回control200の8本のみ新規生成。比較24本は線形ゲインのみ−23.02〜−22.99LUFS、最大−4.22dBTP、サンプル数を保持し、連結PCMも一致。母音伸長・EQ・ノイズ除去・推論重み補正なし。Whisper-smallで今回8本を認識し、本人文の内容一致と他7文の内容保持を補助確認した。数字・句読点・切符の仮名表記差を発音異常とは扱わず、ASRを聴感検証の代わりにしない。

Mac成果物は`output/talk_full_20260917/controlled_retrain_20261023/control200_comparison/`。`listening/user_180_candidate200.wav`は180→今回control200、`user_all_three.wav`は180→従来再学習200→今回control200、各1秒間隔。他7文も同じ構成。隔離エンジン50033は生成後終了、本番アプリと公開ZIPは変更なし。

転送ZIPは373255292bytes、SHA256 `cf1c0badd18ec47157bf9ea66997b556d98afdecbe7d2004aa90f5afe17311a6`。64MiB×5と最終37710972bytesの6部を全size/SHA照合して結合した。Colab直接取得の一部待機をDriveフォルダ`MYCOEIROINK_WORK/tokine_control200_parts`からの取得で補い、canonical全6部を照合。Drive限定で一時変更した自動ダウンロードとポップアップ許可は両方デフォルトへ復元し、実画面を確認した。

再現用`tools/controlled_200_compare.py`はZIP/全モデルhash、checkpoint監査、47音素順序、1049キー・形状・有限値を確認し、既存の隔離runtimeだけに登録する。`--root <比較先> --previous <retrain_comparison_20261023> --cache <pilot_comparison> --bundle <ZIP> --manifest <転送manifest>`で今回の実モデル・全24測定・比較音声まで実行済み。quarter_lr200も同形式のZIPとmanifestを別rootへ渡せる。元の180/181/182の48生成は再実行しない。

2026-10-24 本人が「2epochでは比較する意味がない、進めて」と指示。従前の短期判定による延長保留を取り消し、controlとquarter_lrの両条件を保存済み182から200まで継続する。`bash tools/controlled_start.sh "$pilot" 200`を使用する。第2引数省略は従来どおり182。runの`--max-epoch 200`は実行用設定の終了epochだけを変え、元train.yaml、重み、optimizer、LR系列、損失、録音は維持する。既存のoptimizer stepとschedulerを検査して復元し、各epochを保持する。旧training_complete.jsonのepoch182を200完了と取り違えず、最新checkpoint内部epochとbatchログを確認する。以下の182比較結果は短期観測であり、長期結果の代用にしない。

2026-10-24、両条件の完全checkpointを旧torchでロードし、内部epoch182、G/D全942/106 optimizer状態のstep182000、180/181/182個別モデルとtraining_complete.jsonを確認した。quarter_lrの読込障害からの復旧も成功し、対照2条件の学習部分は完了。新ホストでの復旧launcherも完了済みcheckpointを検出して両条件をスキップした。個別モデルの回収と通常8文×6条件48音声の未補正生成・測定・ASRを完了した。通常LRの182は周期ピークが6/8文で低下したが、高域全体の成分は7/8文で増加。LRを¼にした条件にも一貫した改善はなく、180を超える総合音質や劣化原因の特定はまだ確認できていない。下記の比較結果を参照。最新のホスト・再開情報は共有Obsidianを参照する。

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

## 181〜182の対照実験の比較結果（2026-10-24）

原本180、従来の未補正再学習200、今回のcontrol/quarter_lr各181/182を、隔離COEIROINKの同じ設定で通常8文ずつ生成した。[測定値とASR全文](pilot_comparison.json)。元ZIP全体のSHAと各モデルのSHA・1049キー/形状/有限値・音素順序を照合し、推論重み補正なしで使用。今回の修正実験は182までであり、新たな200epochを完成させたものではない。

| モデル | 周期ピーク指標平均 dB | 母音の8〜20kHz/300〜4000Hz比 dB | 元音声RMS dBFS |
|---|---:|---:|---:|
| 原本180 | 0.265 | −28.637 | −30.136 |
| 従来の再学習200 | 0.473 | −28.563 | −30.206 |
| control181 | 0.403 | −28.097 | −30.728 |
| control182 | 0.059 | −28.049 | −30.477 |
| quarter_lr181 | 0.196 | −28.570 | −30.309 |
| quarter_lr182 | 0.275 | −28.445 | −30.247 |

通常LRの182は周期ピーク指標が180より6/8文で低く、平均も下がった。ただし高域成分比は7/8文で上がり、平均差は+0.588dB。quarter_lr182は周期ピークが4/8文で低下、4/8文で上昇し、平均は180より少し高い。高域成分比も6/8文で上昇した。学習率を下げれば直るという仮説を、この結果は支持しない。ただし各条件1回の学習であり、再現性や長期学習の効果までは判定できない。

周期ピーク指標は符号付きの局所スペクトル差で、ノイズ量やSNRではない。高域成分には声の倍音も含まれる。したがって「周期ピークが下がった」「高域成分が増えた」を、それだけで音質改善・劣化と断定しない。現段階の判定は、**180を超える総合音質の改善は未確認**。聴感による発音・声質・ざらつきの確認も未実施で、検証済みとは扱わない。

全48WAVは44.1kHz/16bit/mono、PCMクリップ0。同音量比較48本と連結24本は線形ゲインだけを使い、−23.02〜−22.98LUFS、最大−4.22dBTP。サンプル数を保持し、母音伸長・EQ・ノイズ除去なし。本人文の比較は `pilot_comparison/listening/user_epoch182_control_quarter.wav`、180→control182→quarter_lr182の順で1秒間隔。`user_all_six.wav`は表の順。公開モデルや本番アプリは変更していない。

Whisper-smallで48音声を認識した。本人文は句読点・空白を除いて6条件とも入力と一致。一方、旅行文の「切符」は条件により「きっと」と認識され、quarter_lrでは「駅前」「公園」も別語になった。音声認識の誤りと実際の発音崩れを分離しておらず、ASRだけで発音改善とは評価しない。初回の認識は既存uv環境のSciPy共有ライブラリ読込エラーで停止したため、scipy.signalとWhisperクラスのimportを確認した別の既存uv環境vScMcImLws_ytlzQへ切替え、全48件の出力を確認した。学習環境の変更や追加インストールはしていない。

182比較直後には延長を保留し別条件の切り分けを検討したが、その後の本人指示により、現在は両条件を同じ設定のまま200まで継続している。低LRを改善策として採用済みという意味ではない。別損失やG/D更新条件への切替えは行わず、まず200までの結果を評価する。過学習や特定設定の誤りが原因との結論は出ていない。

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

## 回収済み候補の通常8文検査（2026-10-25）

`tools/ranking_candidate_survey.py` は既存のSHA棚卸しJSON、ローカルの`<epoch>epoch.pth`、隔離runtimeを受け取る。`--root` は系列ごとに固定し、`--previous` は`retrain_comparison_20261023`、`--inventory` は棚卸しJSON、`--models` は回収先、`--series` は`previous_retrain` / `control` / `quarter_lr`を指定する。ローカルにないepochは処理せず、181/182と200は既存cacheがあるため生成対象から除く。全候補完了の判定はこのスクリプトのcompleteだけで行わず、全系列棚卸しとの突合で行う。

モデルのsize/SHA・1049テンソルのkey/shapeと有限値を180原本に照合する。隔離50033が使用中なら起動せず、終了時に自身のengineだけを停止する。本番50032は変更しない。再開時は既存WAV/JSONの対、本文・速度・style・モデルSHA・WAV SHA・engine SHAを照合し、既存音声を再生成しない。44.1kHz/16bit/mono・クリップ0・有声音フレームと主要測定値の有限値も確認する。母音伸長・EQ・除去・推論重み補正はしない。

前回正規再学習189・191〜199の10モデル80音声は測定・ハッシュ照合を完了した。`ranking_previous_validation.json` とOUTPUTの`ranking_previous`を参照。全条件上位5と広帯域ざらつきの評価、聴感確認は未完了。

`tools/ranking_evidence_index.py` は原本81モデル、pilot、両200、185、追加回収候補の一致する通常文cacheを統合し、欠損を明示する。モデルSHA、本文、速度、WAV SHAを照合し、同一モデル・同一文に異なるWAVがあれば停止する。感度検査用の3dB以上の全ピークを既存のSTFT手順で再計算する。`noise_spectral_detail.detail` の既定4dB・最大8本は従来どおりで、感度用だけ引数を変更する。順位はこのindex作成だけでは確定しない。

189はDrive UIから完全checkpointを回収し、内部epoch189、両scheduler189、optimizer942/106・全step189000・有限momentum、1049モデルテンソルのkey/shape/有限値と抽出後完全一致を確認。無補正の抽出モデルを候補に追加した。`checkpoint189_audit.json`に取得ファイルのローカルSHAを保存したが、Drive側の独立したSHAは未取得と明記する。control/quarterの180は原本180と全1049テンソルの完全一致を確認した。保存形式によるファイルSHA差は`ranking_weight_aliases.json`に記録し、別の音質のモデルと解釈しない。

`tools/ranking_validity.py --root <controlled_retrain_20261023> --asr-python <既存Whisper環境のpython> --asr-script <diagnosis_20261008/tokine_asr.py>` は統合indexのWAV SHAを再照合し、既存FFmpegのLUFS/true peak/LRAと既存ローカルWhisper-smallによる内容確認を収集する。音声は外部へ送らず、原音も変更しない。比較済みのtranscripts.jsonはWAV SHAが一致するものだけ再利用する。新規認識は既存スクリプトを一プロセスで実行し、スクリプトSHAと各入力WAV SHAをasr_job.jsonへ保存、認識ごとにrecords.jsonを原子的に保存する。途中終了後は同じコマンドで不足分から再開し、同じ検査の同時起動はファイルロックで拒否する。進行はranking_evidence/validity_launcher.logとvalidity/records.json、終了状態はvalidity/summary.jsonで確認する。ASRは内容の検査に限り、発音や聴感を検証済みとはしない。全体の候補不足とASR不足は別に表示する。

2026-10-25、回収済み99モデル792音声の音量測定とASR出力収集を完了。全WAV SHA不変、認識出力の空欄0、クリップ0、音量値は全て有限、同文180基準のRMS6dB超低下0を確認した。最新summaryをcacheだけで再作成し、180基準のRMS差/フラグ/長さ比を含む。[監査結果](ranking_validity_validation.json)。これは全候補の回収完了や内容レビュー完了を意味しない。旅行文の「切符」は全99候補で表記が異なり、会議文にも漢数字と算用数字の差があるため、単純な文字一致率を発音精度や自動除外条件にしない。旧原本181では複数文に大きい認識差があり、実音声と認識誤りの分離は未実施。残り34モデルが未評価のため、全体上位5は未確定。
