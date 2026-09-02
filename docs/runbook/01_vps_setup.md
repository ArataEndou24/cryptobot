# 手順書 01: サーバー（VPS）の準備

対象: 運用者。この手順はあなたの得意領域なので、要点と判断基準だけ書く。
所要時間の目安: 1〜2 時間。

## 1. VPS の選定基準

| 項目 | 基準 |
|---|---|
| OS | Ubuntu 24.04 LTS |
| CPU / メモリ | 2 vCPU / 4 GB 以上（データ処理で一時的にメモリを使う） |
| ディスク | 40 GB 以上の SSD（研究データは数 GB） |
| リージョン | シンガポールまたは東京。時間軸の関係で距離はほぼ影響しない |
| 料金 | 月 1,000〜2,000 円程度で十分 |
| 候補 | Vultr、Linode、Hetzner、ConoHa、さくらの VPS |

判断基準は「稼働率の実績」と「料金の単純さ」。速度は不要。

## 2. 初期設定（あなたの通常の手順で構いません）

- 一般ユーザーを作り、sudo を付与し、root ログインを禁止する。
- SSH は公開鍵認証のみ。パスワード認証は無効化する。
- ファイアウォール（ufw）で SSH 以外を閉じる。ボットは外向き通信しかしない。
- 自動セキュリティ更新（unattended-upgrades）を有効にする。
- タイムゾーンは UTC にする。ボットの時刻は全て UTC で扱う。

```bash
sudo timedatectl set-timezone UTC
```

## 3. 必要なソフトの導入

```bash
# git と基本ツール
sudo apt update && sudo apt install -y git make curl unzip

# uv（Python の環境管理ツール。Python 本体も自動で入る）
curl -LsSf https://astral.sh/uv/install.sh | sh
# シェルを開き直すか、表示される案内どおりに PATH を通す

# Docker（後の段階で使う。今は入れるだけで良い）
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

## 4. リポジトリの取得と自己診断

```bash
git clone <このリポジトリの URL> ~/cryptobot
cd ~/cryptobot
make setup     # 依存ライブラリを入れ、設定ファイルの雛形をコピーする
make doctor    # 環境診断。全て OK になるまで先に進まない
make check     # 自己診断。「全ての検査に合格しました」と出れば正常
```

`make doctor` で NG が出たら、その出力をそのまま貼り付けて報告してください。

## 5. 研究用データの取得

```bash
make data      # 初回は数十分〜1 時間程度。回線次第
make data-status
make universe  # 現時点の対象銘柄が表示されれば成功
```

2 回目以降の `make data` は差分だけ取得するので短時間で終わります。
毎日 1 回、自動で実行する設定（cron）は後の手順書で扱います。

## 6. やってはいけないこと

- 秘密鍵、API キー、`.env` の中身をチャットや他人に貼らない。
- `config/settings.yaml` と `.env` を git にコミットしない（除外設定済みだが、念のため）。
- 本番（mainnet）への切り替えは、手順書 03 の条件を満たすまで行わない。
