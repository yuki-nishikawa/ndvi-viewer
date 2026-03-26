# 衛星指数 Viewer — Sentinel-2

Sentinel-2 衛星画像から植生指数（NDVI / EVI / NDRE）を可視化する Streamlit アプリです。

## 機能

| 機能 | 説明 |
|------|------|
| エリア選択 | 地図上で矩形を描画して解析対象エリアを指定 |
| シーン検索 | 期間・雲量・取得数でフィルタリングして Sentinel-2 シーンを検索 |
| 雲マスキング | SCL バンドを使った雲・雲影の除去（3段階） |
| 指数計算 | NDVI / EVI / NDRE をピクセル単位で計算 |
| インタラクティブマップ | レイヤー切替・不透明度調整付きの地図表示 |
| 時系列スライダー | 複数シーンを日付順に切り替えて変化を比較 |
| ポイント分析 | 地図クリックで各指数値を取得し、時系列グラフを表示 |
| バンドキャッシュ | ダウンロード済みバンドをローカルにキャッシュして再計算を高速化 |

### 対応指数

| 指数 | 正式名称 | 使用バンド |
|------|----------|-----------|
| NDVI | 植生指数 | Red, NIR |
| EVI | 大気補正強化型植生指数 | Blue, Red, NIR |
| NDRE | レッドエッジ指数 | RedEdge1, NIR08 |

## 動作環境

- Python 3.13
- [mise](https://mise.jdx.dev/) でツールバージョンを管理
- [Poetry](https://python-poetry.org/) で依存関係を管理

## セットアップ

```bash
# mise でツールをインストール（Python 3.13 / Poetry 2.x）
mise install

# 依存関係をインストール
poetry install
```

## 起動

```bash
poetry run streamlit run app.py
```

ブラウザが自動で開き `http://localhost:8501` にアクセスできます。

## 使い方

1. **Step 1 — エリアを選択**: 地図上の矩形ツール（□）でエリアを描画し「このエリアで検索する」をクリック
2. **Step 2 — シーンを検索**: サイドバーで期間・雲量を設定し「シーンを検索」をクリック
3. シーン一覧から解析対象を選択し「選択シーンで指数を計算する」をクリック
4. **Step 4 — 指数マップ**: NDVI / EVI / NDRE をレイヤーで切り替えて確認。地図をクリックするとポイント値と時系列グラフを表示

## データソース

[Earth Search STAC API (Element84)](https://earth-search.aws.element84.com/v1) 経由で `sentinel-2-l2a` コレクションを参照します。認証不要（Public）。

## バンドキャッシュ

ダウンロードしたバンドデータは `~/.cache/ndvi_viewer/` に pickle 形式でキャッシュされます。サイドバーの「キャッシュを削除」ボタンで削除できます。

## 依存ライブラリ

- [Streamlit](https://streamlit.io/)
- [Folium](https://python-visualization.github.io/folium/) + [streamlit-folium](https://folium.streamlit.app/)
- [rasterio](https://rasterio.readthedocs.io/)
- [pystac-client](https://pystac-client.readthedocs.io/)
- NumPy / Matplotlib
