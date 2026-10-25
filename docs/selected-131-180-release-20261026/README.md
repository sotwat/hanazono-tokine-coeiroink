# 131・180epoch配布版 v1.3.0

本人が長文試聴で良いと評価し、登録・配布を2つだけにするよう指定した旧原本131/180を収録。数値順位上位2件という意味ではなく、2つの優劣も未指定。134/145の明確なノイズ指摘を保持する。追加学習・監視停止を維持。

`tools/package_selected_models.py`で原本モデルと既存8文cacheのSHAを照合して構築。モデル/音素設定/画像/各3本の原音サンプルを含み、EQ・ノイズ除去・推論重み補正・音量補正なし。ZIP全CRCと全同梱ファイルSHAを検証。詳細はpackage_validation.json。

Macの同一話者UUIDを1131/1180だけ登録し、他話者は維持。旧登録metadataと上書きするモデルフォルダをOUTPUT/selected131_180_release_20261026/previous_registrationへ退避。旧モデルの物理削除はしていない。実アプリのキャラクター設定で131/180の2件を確認。両スタイルを本番50032で生成し試聴に用いた既存cacheとPCM完全一致、installed_validation.json。

作業中の文章・F0・調声をMusic/CoeiroFX/書き出し/131・180登録前_20261026.cinkへ別名保存し、styleId/styleNameだけ131へ変更した別コピーを作成。元ファイルを保護。

配布パッケージはBASE/配布用パッケージ/hanazono-tokine-coeiroink-131-180epoch-v1.3.0.zip。歴代リリースは履歴として保持。
