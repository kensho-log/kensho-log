# 検証ログ | Finance Brute-Force Lab

[![CI](https://github.com/kensho-log/kensho-log/actions/workflows/ci.yml/badge.svg)](https://github.com/kensho-log/kensho-log/actions/workflows/ci.yml)

投資の通説は、本当に正しいのか。  
過去データを総当たりで計算し、その差分だけを提示する検証ログ。

- YouTube: [@kensho_log](https://www.youtube.com/@kensho_log)
- 方針: 意見ではなく、Python によるシミュレーション結果のみを無機質に公開

---

## 本リポジトリの位置付け

本リポジトリは、YouTube チャンネル **「検証ログ | Finance Brute-Force Lab」** で公開する全検証動画に対応する、検証コード・データセット・単体テスト・生成済み数値サマリの一次情報源である。

- 採否基準を公開する
- 検証コードを全公開する
- データソースと取得日時を全公開する
- 計算ミスが発覚した場合は訂正ログを第一級コンテンツとして公開する

視聴者は本リポジトリの該当コミットハッシュ（動画内字幕に焼き込み）をチェックアウトすれば、動画の全数値を再現できる。

---

## 免責事項

本リポジトリおよび対応する動画は、過去データに対する計算結果の記録である。  
将来の投資成果を示唆するものではなく、投資助言ではない。  
投資判断は各自の責任で行うこと。

特定の金融商品・証券会社・制度の推奨は行わない。  
「買うべき」「おすすめ」「最適解」「勝てる」「稼げる」の語は、コード内コメント・README・動画・Issue を含めて使用しない。

---

## 採否基準（動画化判定）

検証スクリプト実行後の差分 % により、以下の通り機械的に判定する。判定は主観を介さず、CI 上で自動付与する。

| 条件              | 動画化方針                              |
| ----------------- | --------------------------------------- |
| 差分 >= 10%       | 長尺メイン                              |
| 差分 5 - 10%      | Shorts                                  |
| 差分 < 5%         | リポジトリログのみ（動画化せず）        |
| 直感通りの結果    | Shorts 送り（公開必須・選別バイアス排除）|

選別バイアスを避けるため、「差分が小さかった検証」「直感通りだった検証」も GitHub には必ず残す。

---

## 計算信頼性の多層防衛

- データソース二重化（例: `yfinance` + `pandas-datareader` (Stooq) 等、差異 > 0.1% で警告停止）
- `pytest` による配当再投資・リバランス・税計算の単体テスト
- GitHub Actions で push ごとに CI 実行
- 動画生成時に `git rev-parse HEAD` でコミットハッシュを取得 → 映像字幕に焼き込み
- 動画末尾および本リポジトリ `docs/LIMITATIONS.md` に「本検証の限界」を固定テンプレで記載

---

## ディレクトリ構成

```
kensho-log/
├─ .cursor/rules/        # Cursor 用プロジェクトルール
├─ .github/              # GitHub Actions (CI)
├─ data/
│   ├─ raw/              # 外部取得データキャッシュ（.gitignore）
│   └─ processed/        # 加工済みデータ（.gitignore）
├─ docs/                 # 方法論ドキュメント
├─ notebooks/            # 探索用（コミット除外）
├─ output/
│   ├─ figures/          # matplotlib 生成 PNG
│   └─ videos/           # FFmpeg 生成動画
├─ scripts/              # 動画生成バッチ等
├─ src/                  # 検証ロジック本体
├─ tests/                # pytest 単体テスト
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## 開発環境

- Windows / メモリ 8GB ノートPC運用前提
- Python 3.12
- FFmpeg（`subprocess` 直接呼び出し）
- MoviePy 等の重量ライブラリは使用禁止

### セットアップ

```powershell
cd C:\Users\yoshioka2024\AppData\Local\kensho-log
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 単体テスト

```powershell
.\.venv\Scripts\Activate.ps1
pytest
```

---

## 公開物

- GitHub（本リポジトリ）: 全検証コード・データセット・単体テスト
- Zenn: 方法論ホワイトペーパー
- X: 次回検証テーマ募集と議論

---

## ライセンス

（未設定。公開前に決定する）
