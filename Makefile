# 運用者が使うコマンド一覧。`make` だけを打つとこの一覧が出ます。
.DEFAULT_GOAL := help

help: ## コマンド一覧を表示
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## 初回セットアップ（依存ライブラリの導入）
	uv sync
	@test -f config/settings.yaml || cp config/settings.example.yaml config/settings.yaml
	@echo "完了。次は 'make doctor' で環境診断をしてください。"

doctor: ## 環境診断（ネットワーク、設定、ディスク）
	uv run cryptobot doctor

check: ## 自己診断（全ての検査を実行。緑なら正常）
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest -q
	@echo "全ての検査に合格しました。"

fmt: ## コード整形（開発者用）
	uv run ruff format .
	uv run ruff check --fix .

data: ## 研究用データの取得と更新（時間がかかります）
	uv run cryptobot data download

data-status: ## 取得済みデータの状態を表示
	uv run cryptobot data status

universe: ## 現時点の対象銘柄（ユニバース）を表示
	uv run cryptobot universe show

backtest: ## 戦略を過去データで検証して成績を表示
	uv run cryptobot backtest run

walkforward: ## ウォークフォワード検証（過学習の検出。数分かかります）
	uv run cryptobot backtest walkforward

compare: ## 戦略の構成要素ごとの比較（どこに優位性があるかを見る）
	uv run cryptobot backtest compare

exchange-symbols: ## 取引所（Hyperliquid）の上場銘柄一覧を取得して保存
	uv run cryptobot exchange symbols

plan: ## 今この時点の目標ポジションを表示（注文は出さない）
	uv run cryptobot live plan

.PHONY: help setup doctor check fmt data data-status universe backtest walkforward compare exchange-symbols plan
