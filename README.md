# cryptobot

暗号資産の自動売買システム。研究（過去データでの検証）、練習（テストネット）、本番運用を
一つの基盤で行う。

**運用者はコードを読まなくてよい。** 触るのは設定ファイル 1 枚とコマンド数個だけ。

## まず読むもの

- `docs/runbook/01_vps_setup.md` サーバーの準備
- `docs/runbook/02_hyperliquid_wallet.md` ウォレットと取引所の準備
- `docs/decisions.md` なぜこう設計したかの記録
- `docs/architecture.md` 全体構成

## コマンド

`make` と打つと一覧が出る。

| コマンド | 役割 |
|---|---|
| `make setup` | 初回セットアップ |
| `make doctor` | 環境診断。NG があれば出力を貼って報告 |
| `make check` | 自己診断。「全ての検査に合格しました」なら正常 |
| `make data` | 研究用データの取得と更新 |
| `make data-status` | 取得済みデータの状態 |
| `make universe` | 現時点の対象銘柄 |

## 設定

`config/settings.yaml`（`make setup` で雛形からコピーされる）。各項目に日本語の説明がある。
秘密鍵などは `.env` に書く（`.env.example` を参照）。どちらも git には含まれない。

## 現在の状態

データ層（Binance 配布データの取得、保存、ユニバース選定）まで実装済み。
検証エンジン、戦略、執行、監視は今後の実装。
