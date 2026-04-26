# Character Asset — ログ

検証ログの解説キャラクター「ログ」の立ち絵素材を格納する。

## 設計方針

- **独自 IP**：VOICEVOX 四国めたん の音声と組み合わせるが、ビジュアルは独立した別キャラクター。
- **トーン**：チャンネル方針「ダーク・無機質・ローファイ」に合わせる。
- **用途**：動画右 pane に固定配置、常時表示。
- **クレジット表記**：音声は動画フッタに `Voice: VOICEVOX / 四国めたん` を常時表示。立ち絵にはクレジット不要（独自生成物）。

## 必須ファイル

| ファイル名 | 用途 | 必要度 |
|---|---|---|
| `log_neutral.png` | 中立表情・常時使用 | **必須** |
| `log_explain.png` | 説明中の表情（手振り等）| 任意（長尺版で使用） |
| `log_concern.png` | 暴落局面の表情 | 任意（長尺版で使用） |
| `log_reveal.png` | フィナーレ時の表情 | 任意（長尺版で使用） |
| `log_mouth_open.png` | 口パク用（口開きバリエーション）| 任意（MVP は静止でも可） |

MVP では `log_neutral.png` の 1 枚のみで十分。

## 画像仕様

| 項目 | 値 |
|---|---|
| フォーマット | PNG（透過アルファ必須） |
| 解像度 | 長辺 1024 px 以上推奨（動画内では 300×500 程度にスケール） |
| 縦横比 | 縦長（2:3 〜 9:16）の肖像構図 |
| 背景 | 完全透過（alpha channel） |
| 色空間 | sRGB |
| ファイルサイズ | 1 枚あたり 2 MB 以下目安 |

## 生成手順

### 1. Bing Image Creator で候補生成

- URL: https://www.bing.com/create
- 既存 Microsoft アカウントでログイン
- プロンプト（英語、コピペ推奨）:

```
professional anime-style woman financial analyst,
late 20s, calm neutral expression, mature atmosphere,
wearing dark navy suit or black turtleneck,
portrait framing chest up,
dark neutral background,
soft rim light, clean line art, muted color palette,
analytical serious mood, minimal decoration
```

- 3〜5 枚生成し、トーンに最も合う 1 枚を選定
- ダウンロードして一時保存

### 2. 背景透過処理

- URL: https://www.remove.bg/
- アカウント不要、1 回無料（アップロード → PNG ダウンロード）
- 透過 PNG として保存

### 3. リポジトリに配置

```
assets/character/log_neutral.png
```

### 4. Git commit

```powershell
git add assets/character/log_neutral.png assets/character/README.md
git commit -m "assets(character): add log_neutral.png for hypothesis 001 narration"
```

## ライセンス

- Bing Image Creator 生成物：Microsoft Services Agreement に基づき、生成者に商用利用権あり（2026 年時点）
- remove.bg 無料利用：生成物の商用利用可
- 本プロジェクト（検証ログ）内では、本リポジトリ所有者が著作権を保有する扱いとする

## 将来の拡張

- **Live2D 化**：ログの 2D イラストを Live2D Cubism でモデリングし、口パク・瞬きを動的制御。ただし Cubism は 8GB RAM で動作するが学習コスト高。Phase D 以降。
- **表情差分**：長尺メイン版（9 分）向けに `log_explain`, `log_concern`, `log_reveal` を追加生成。AI 画像の一貫性確保のため、同一 seed + 同一プロンプト + 表情指示のみ差し替えで運用する。
