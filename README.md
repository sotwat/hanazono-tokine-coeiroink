# 花園トキネ COEIROINK（通常版 100epoch）

テキスト読み上げソフトウェア「COEIROINK」向けの合成音声モデル（MYCOEIROINK）です。
本リポジトリでは、100epoch学習モデルおよびアイコン・立ち絵・音声を梱包した通常版パッケージを公開しています。

## 概要

- **話者名**: 花園トキネ
- **スタイル**: ノーマル（100epoch）
- **バージョン**: 2.1.0
- **形式**: COEIROINK v2 形式

## ダウンロード

最新の配布パッケージ（ZIP）は [Releases](https://github.com/sotwat/hanazono-tokine-coeiroink/releases/latest) よりダウンロードいただけます。

- **ファイル名**: `hanazono-tokine-coeiroink-100epoch.zip`

## 導入方法（COEIROINK v2）

### 方法1: アプリ画面から簡単追加（推奨）
1. ダウンロードしたZIPファイルを解凍（展開）します。
2. COEIROINK（v2）を起動します。
3. 画面左上のメニュー または 設定アイコンから「話者の追加」を開きます。
4. 「フォルダを選択」をクリックし、解凍された `hanazono-tokine-2.1.0` フォルダを選択します。
   （※ZIPドラッグ＆ドロップに対応している環境では、ZIPをそのままアプリへドロップでも追加可能です）
5. 話者一覧に「花園トキネ」が表示されれば導入完了です。

### 方法2: フォルダへ直接配置する場合
1. 解凍してできた `hanazono-tokine-2.1.0` フォルダを、COEIROINKの `speaker_info` フォルダ内に配置します。
   - **Windowsの場合**:
     COEIROINK実行ファイルと同じ階層にある `speaker_info` フォルダ内
   - **Macの場合**:
     `COEIROINKv2.app` を右クリック →「パッケージの内容を表示」→ `Contents/MacOS/speaker_info/` フォルダ内
2. COEIROINKを再起動すると「花園トキネ」が使用可能になります。

## 利用規約・クレジット表記

本音声ライブラリを使用して作成した音声を公開・配信される際は、以下のクレジット表記をお願いいたします。

```text
COEIROINK:花園トキネ
```
または
```text
音声：花園トキネ (COEIROINK)
```

詳細な利用規約（商用・非商用利用、禁止事項等）については、同封の `policy.md` を必ずご確認ください。

## 関連リンク
- COEIROINK 公式サイト: https://coeiroink.com/
- 花園トキネDS（歌声音源）: https://github.com/sotwat/hanazono-tokine-ds
