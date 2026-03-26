import hashlib
import pickle
import time
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import folium
import folium.plugins
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import streamlit as st
from matplotlib.colors import Normalize
from pystac_client import Client
from streamlit_folium import st_folium
from rasterio.crs import CRS
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BAND_CACHE_DIR = Path.home() / ".cache" / "ndvi_viewer"

# macOS の日本語フォントを優先して使用する
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Maru Gothic Pro", "AppleGothic", "sans-serif"]

# ---------- 指数設定 ----------

def calc_ndvi(red, nir):
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, np.nan, (nir - red) / denom)


def calc_evi(blue, red, nir):
    denom = nir + 6 * red - 7.5 * blue + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, np.nan, 2.5 * (nir - red) / denom)


def calc_ndre(re, nir):
    denom = nir + re
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom == 0, np.nan, (nir - re) / denom)


# asset_keys: {内部キー: STACアセット名}
INDEX_CONFIG = {
    "NDVI": {
        "label": "NDVI（植生指数）",
        "asset_keys": {"red": "red", "nir": "nir"},
        "calc": lambda b: calc_ndvi(b["red"], b["nir"]),
        "vmin": -0.2,
        "vmax": 1.0,
    },
    "EVI": {
        "label": "EVI（大気補正強化型植生指数）",
        "asset_keys": {"blue": "blue", "red": "red", "nir": "nir"},
        "calc": lambda b: calc_evi(b["blue"], b["red"], b["nir"]),
        "vmin": -0.2,
        "vmax": 1.0,
    },
    "NDRE": {
        "label": "NDRE（レッドエッジ指数）",
        "asset_keys": {"re": "rededge1", "nir": "nir08"},
        "calc": lambda b: calc_ndre(b["re"], b["nir"]),
        "vmin": -0.2,
        "vmax": 1.0,
    },
}

# ---------- コアロジック ----------

def _band_cache_path(asset_href: str, bbox: tuple) -> Path:
    key = hashlib.sha256(f"{asset_href}|{bbox}".encode()).hexdigest()
    return BAND_CACHE_DIR / f"{key}.pkl"


def _cache_stats() -> tuple[int, float]:
    """(ファイル数, MB) を返す。ディレクトリ未作成時は (0, 0.0)。"""
    if not BAND_CACHE_DIR.exists():
        return 0, 0.0
    files = list(BAND_CACHE_DIR.glob("*.pkl"))
    return len(files), sum(p.stat().st_size for p in files) / 1024 / 1024


@st.cache_data(ttl=3600, show_spinner=False)
def search_items(bbox: tuple, date_start: str, date_end: str, cloud_cover_max: int, max_items: int):
    """STAC 検索。同じ条件なら1時間キャッシュ。"""
    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{date_start}/{date_end}",
        query=[f"eo:cloud_cover<{cloud_cover_max}"],
        max_items=max_items,
    )
    return list(search.items())


def read_band_subset(asset_href, bbox):
    BAND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _band_cache_path(asset_href, bbox)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # S3 上の COG を効率的に読むための GDAL 設定
    gdal_env = rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_MULTIPLEX="YES",
        GDAL_HTTP_VERSION=2,
    )

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with gdal_env, rasterio.open(asset_href) as src:
                west, south, east, north = transform_bounds("EPSG:4326", src.crs, *bbox)
                clipped = (
                    max(west, src.bounds.left),
                    max(south, src.bounds.bottom),
                    min(east, src.bounds.right),
                    min(north, src.bounds.top),
                )
                if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                    raise RuntimeError("BBoxがラスタの範囲外です。別のシーンを選択してください。")
                window = from_bounds(*clipped, transform=src.transform)
                window = window.round_offsets().round_lengths()
                if window.width == 0 or window.height == 0:
                    raise RuntimeError("BBoxがラスタに重複しません。別のシーンを選択してください。")
                arr = src.read(1, window=window).astype(np.float32)
                win_transform = src.window_transform(window)
                crs = src.crs
            break  # 成功
        except RuntimeError:
            raise  # BBox 起因のエラーはリトライしない
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s → 2s
    else:
        raise RuntimeError(f"バンドの読み込みに3回失敗しました: {last_err}")

    result = (arr, win_transform, crs)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def scale_reflectance(arr):
    arr = arr / 10000.0
    arr[arr < 0] = np.nan
    return arr


_MAX_GRID_DIM = 1024


def _compute_target_grid(bbox):
    """bbox に対する共通 WGS84 出力グリッドを決定する（10m 相当）。"""
    bw, bs, be, bn = bbox
    center_lat = (bs + bn) / 2
    m_per_deg_lat = 111320
    m_per_deg_lng = 111320 * np.cos(np.radians(center_lat))
    cols = max(1, round((be - bw) * m_per_deg_lng / 10))
    rows = max(1, round((bn - bs) * m_per_deg_lat / 10))
    if max(rows, cols) > _MAX_GRID_DIM:
        scale = _MAX_GRID_DIM / max(rows, cols)
        cols = max(1, round(cols * scale))
        rows = max(1, round(rows * scale))
    return transform_from_bounds(bw, bs, be, bn, cols, rows), (rows, cols)


# 雲マスク強度ごとの SCL 除外クラス定義
# SCL: 1=欠陥, 3=雲影, 8=中確度雲, 9=高確度雲, 10=薄雲(巻雲)
_CLOUD_MASK_LEVELS = {
    "高確度のみ":   frozenset({9, 10}),        # データ欠損が最小
    "標準（推奨）": frozenset({8, 9, 10}),     # 中確度以上
    "積極的":       frozenset({1, 3, 8, 9, 10}), # 雲影・欠陥も除外
}


def _compute_cloud_mask(item, bbox: tuple, dst_transform, dst_shape, scl_classes: frozenset):
    """SCL バンドから雲マスク (True=雲/雲影) を作成する。取得失敗時は None。"""
    try:
        scl_href = item.assets["scl"].href
        scl_raw, src_transform, src_crs = read_band_subset(scl_href, bbox)
        dst = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=scl_raw, destination=dst,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.nearest,
            src_nodata=0, dst_nodata=0,
        )
        return np.isin(dst.astype(np.int32), list(scl_classes))
    except Exception:
        return None


def compute_index(item_id, hrefs: dict, bbox: tuple, index_name: str):
    """各バンドを共通 WGS84 グリッドに個別投影してから指数を計算する。"""
    cfg = INDEX_CONFIG[index_name]
    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_shape = _compute_target_grid(bbox)

    bands = {}
    for key, href in hrefs.items():
        arr, src_transform, src_crs = read_band_subset(href, bbox)
        arr = scale_reflectance(arr)
        dst = np.full(dst_shape, np.nan, dtype=np.float32)
        reproject(
            source=arr, destination=dst,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
        bands[key] = dst

    result = cfg["calc"](bands)
    result = np.where(np.abs(result) > 2, np.nan, result)
    bw, bs, be, bn = bbox
    return result, (bw, bs, be, bn)


# ---------- UI ヘルパー ----------

def extract_pixel_value(arr, geo_bounds, lat, lng):
    """geo_bounds=(W,S,E,N) の配列から (lat, lng) の値を取得する。"""
    west, south, east, north = geo_bounds
    nrows, ncols = arr.shape
    if not (west <= lng <= east and south <= lat <= north):
        return np.nan
    col = max(0, min(int((lng - west) / (east - west) * ncols), ncols - 1))
    row = max(0, min(int((north - lat) / (north - south) * nrows), nrows - 1))
    return float(arr[row, col])


def _extract_bbox_from_drawing(map_data):
    """st_folium の返値から矩形を [W, S, E, N] に変換する。見つからない場合は None。"""
    drawing = (map_data or {}).get("last_active_drawing")
    if not drawing:
        all_d = (map_data or {}).get("all_drawings") or []
        drawing = all_d[-1] if all_d else None
    if not drawing:
        return None
    coords = drawing.get("geometry", {}).get("coordinates", [[]])[0]
    if not coords:
        return None
    lngs = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lngs), min(lats), max(lngs), max(lats)]


def make_draw_map(initial_bbox=None):
    """Step 1 用: 矩形描画専用 Folium マップを返す。"""
    m = folium.Map(location=[35.0, 136.0], zoom_start=5, tiles="OpenStreetMap")
    folium.plugins.Draw(
        draw_options={
            "rectangle": True,
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": False, "remove": False},
    ).add_to(m)
    if initial_bbox:
        w, s, e, n = initial_bbox
        folium.Rectangle([[s, w], [n, e]], color="blue", fill=True,
                         fill_opacity=0.1).add_to(m)
        m.fit_bounds([[s, w], [n, e]])
    return m


def reset_all():
    """セッションステートをリセットして Step 1 に戻る。"""
    for k in ["bbox", "items", "selected_item_ids", "computed", "_prev_sid", "clicked_point"]:
        st.session_state.pop(k, None)
    st.session_state["step"] = 1


def _run_computation(items, bbox_tuple, scl_classes: frozenset | None = None):
    """全シーン × 全指数を計算して session_state["computed"] に保存する。"""
    computed = {}
    total = len(items) * len(INDEX_CONFIG)
    progress = st.progress(0, text="計算中...")
    done = 0

    # 雲マスクをシーンごとに事前計算（SCL バンドはキャッシュ済みなら高速）
    cloud_masks: dict = {}
    if scl_classes:
        dst_transform, dst_shape = _compute_target_grid(bbox_tuple)
        for item in items:
            cloud_masks[item.id] = _compute_cloud_mask(
                item, bbox_tuple, dst_transform, dst_shape, scl_classes
            )

    for item in items:
        computed[item.id] = {}
        for index_name, cfg in INDEX_CONFIG.items():
            try:
                hrefs = {k: item.assets[ak].href for k, ak in cfg["asset_keys"].items()}
                arr, geo_bounds = compute_index(item.id, hrefs, bbox_tuple, index_name)
                mask = cloud_masks.get(item.id)
                if mask is not None:
                    arr = arr.copy()
                    arr[mask] = np.nan
                computed[item.id][index_name] = (arr, geo_bounds)
            except Exception as e:
                computed[item.id][index_name] = str(e)
            done += 1
            progress.progress(done / total)
    progress.empty()
    st.session_state["computed"] = computed


# ---------- 描画ユーティリティ ----------

def index_to_rgba(arr, vmin, vmax):
    """指数配列を RdYlGn RGBA uint8 に変換。NaN は透明。"""
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = matplotlib.colormaps["RdYlGn"](norm(np.where(np.isnan(arr), 0, arr)))
    rgba[np.isnan(arr), 3] = 0.0
    return (rgba * 255).astype(np.uint8)


def render_folium_map(computed_scene: dict, bbox: tuple, opacity: float) -> folium.Map:
    """1シーン分の全指数を FeatureGroup + LayerControl で1枚の地図に描画する。"""
    m = folium.Map(tiles="OpenStreetMap")
    m.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])
    for i, (index_name, cfg) in enumerate(INDEX_CONFIG.items()):
        data = computed_scene.get(index_name)
        if not isinstance(data, tuple):
            continue
        arr, geo_bounds = data
        geo_west, geo_south, geo_east, geo_north = geo_bounds
        fg = folium.FeatureGroup(name=cfg["label"], show=(i == 0))
        folium.raster_layers.ImageOverlay(
            image=index_to_rgba(arr, cfg["vmin"], cfg["vmax"]),
            bounds=[[geo_south, geo_west], [geo_north, geo_east]],
            opacity=opacity,
        ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_index_figure(arr, geo_bounds, vmin, vmax, index_name, title):
    geo_west, geo_south, geo_east, geo_north = geo_bounds
    extent = (geo_west, geo_east, geo_south, geo_north)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, extent=extent, vmin=vmin, vmax=vmax, cmap="RdYlGn", origin="upper", aspect="auto")
    fig.colorbar(im, ax=ax, label=index_name)
    ax.set_title(title)
    ax.set_xlabel("経度")
    ax.set_ylabel("緯度")
    fig.tight_layout()
    return fig


# ---------- Streamlit UI ----------

st.set_page_config(page_title="衛星指数 Viewer", layout="wide")
st.title("衛星指数 Viewer — Sentinel-2")

if "step" not in st.session_state:
    st.session_state["step"] = 1

# ===== サイドバー =====
with st.sidebar:
    st.header("検索条件")
    date_start = st.date_input("開始日", value=date.today() - timedelta(days=30))
    date_end = st.date_input("終了日", value=date.today())
    cloud_cover_max = st.slider("雲量上限 (%)", 0, 100, 20)
    max_items = st.number_input("最大取得シーン数", min_value=1, max_value=30, value=5)

    st.divider()
    _mask_options = ["なし"] + list(_CLOUD_MASK_LEVELS.keys())
    _mask_sel = st.selectbox("雲マスキング（SCL）", _mask_options, index=2,
                             help="高確度のみ: SCL 9+10 / 標準: 8+9+10 / 積極的: 1+3+8+9+10")
    cloud_scl_classes = _CLOUD_MASK_LEVELS.get(_mask_sel)  # None if "なし"

    st.divider()
    st.subheader("バンドキャッシュ")
    n_files, size_mb = _cache_stats()
    st.caption(f"保存先: `{BAND_CACHE_DIR}`")
    st.caption(f"使用量: {size_mb:.1f} MB（{n_files} ファイル）")
    if st.button("キャッシュを削除", use_container_width=True, disabled=n_files == 0):
        for p in BAND_CACHE_DIR.glob("*.pkl"):
            p.unlink()
        st.success("キャッシュを削除しました。")
        st.rerun()

    if st.session_state.get("step", 1) > 1:
        st.divider()
        st.button("最初からやり直す", on_click=reset_all, use_container_width=True)

# ===== Step 1: エリア選択 =====
st.header("Step 1: エリアを選択")
st.info("地図上で矩形ツール（□）を使ってエリアを描画し、確定ボタンを押してください。")

col_reset, _ = st.columns([1, 4])
if col_reset.button("描画をリセット", use_container_width=True):
    st.session_state["draw_map_ver"] = st.session_state.get("draw_map_ver", 0) + 1
    st.rerun()

map_data_step1 = st_folium(
    make_draw_map(st.session_state.get("bbox")),
    key=f"step1_map_{st.session_state.get('draw_map_ver', 0)}",
    height=450,
    use_container_width=True,
    returned_objects=["all_drawings", "last_active_drawing"],
)
pending_bbox = _extract_bbox_from_drawing(map_data_step1)
if pending_bbox:
    w, s, e, n = pending_bbox
    st.success(f"選択エリア: West={w:.4f}, South={s:.4f}, East={e:.4f}, North={n:.4f}")
    if st.button("このエリアで検索する →", type="primary"):
        st.session_state["bbox"] = pending_bbox
        st.session_state["step"] = 2
        st.rerun()

# ===== Step 2: シーン検索・選択 =====
if st.session_state.get("step", 1) >= 2:
    st.divider()
    st.header("Step 2: シーンを検索・選択")
    bbox = st.session_state["bbox"]
    st.caption(f"エリア: West={bbox[0]:.4f}, South={bbox[1]:.4f}, East={bbox[2]:.4f}, North={bbox[3]:.4f}")

    if st.button("シーンを検索", type="primary"):
        for k in ["items", "selected_item_ids", "computed", "_prev_sid",
                  "clicked_NDVI", "clicked_EVI", "clicked_NDRE"]:
            st.session_state.pop(k, None)
        if date_start >= date_end:
            st.error("開始日は終了日より前に設定してください。")
            st.stop()
        with st.spinner("検索中..."):
            try:
                found = search_items(
                    tuple(bbox), date_start.isoformat(), date_end.isoformat(),
                    int(cloud_cover_max), int(max_items),
                )
            except Exception as e:
                st.error(f"検索エラー: {e}")
                st.stop()
        st.session_state["items"] = found
        st.rerun()

    if "items" in st.session_state:
        items = st.session_state["items"]
        if not items:
            st.warning("条件に一致するシーンが見つかりません。期間・雲量を調整してください。")
        else:
            st.success(f"{len(items)} シーン見つかりました。")
            scene_labels = [
                f"{i.properties.get('datetime', '')[:10]}  雲量:{i.properties.get('eo:cloud_cover', 0):.1f}%  [{i.id[:24]}]"
                for i in items
            ]
            selected_labels = st.multiselect(
                "計算するシーンを選択（複数可）",
                scene_labels,
                default=scene_labels[:1],
            )
            if selected_labels:
                if st.button("選択シーンで指数を計算する →", type="primary"):
                    id_map = {lbl: item for lbl, item in zip(scene_labels, items)}
                    sel_items = [id_map[lbl] for lbl in selected_labels]
                    st.session_state["selected_item_ids"] = [i.id for i in sel_items]
                    _run_computation(sel_items, tuple(bbox), scl_classes=cloud_scl_classes)
                    st.session_state["step"] = 4
                    st.rerun()

# ===== Step 4: 指数マップ =====
if st.session_state.get("step", 1) >= 4:
    st.divider()
    st.header("Step 4: 指数マップ")

    computed = st.session_state["computed"]
    items_all = st.session_state["items"]
    bbox = st.session_state["bbox"]

    # 日時順ソート
    scene_ids = sorted(
        computed.keys(),
        key=lambda sid: next(
            (i.properties.get("datetime", "") for i in items_all if i.id == sid), ""
        ),
    )

    # 時系列スライダー（複数シーンのみ）
    def _scene_date(sid):
        it = next((i for i in items_all if i.id == sid), None)
        return it.properties.get("datetime", "")[:10] if it else sid

    if len(scene_ids) > 1:
        current_sid = st.select_slider(
            "時系列スライダー",
            options=scene_ids,
            format_func=_scene_date,
            key="ts_slider",
        )
    else:
        current_sid = scene_ids[0]

    current_item = next(i for i in items_all if i.id == current_sid)
    col_dt, col_cc = st.columns(2)
    col_dt.metric("撮影日時", current_item.properties.get("datetime", "")[:10])
    col_cc.metric("雲量", f"{current_item.properties.get('eo:cloud_cover', 0):.1f}%")

    # シーン変化時にクリックポイントをリセット
    if current_sid != st.session_state.get("_prev_sid"):
        st.session_state.pop("clicked_point", None)
        st.session_state["_prev_sid"] = current_sid

    # ===== 単一マップ（全指数を FeatureGroup で重畳）=====
    opacity = st.slider("不透明度", 0.1, 1.0, 0.7, 0.05, key="opacity_global")
    map_out = st_folium(
        render_folium_map(computed[current_sid], bbox, opacity),
        key=f"map_{current_sid}",
        height=520,
        use_container_width=True,
        returned_objects=["last_clicked"],
    )
    clicked = (map_out or {}).get("last_clicked")
    if clicked:
        st.session_state["clicked_point"] = clicked

    # ===== ポイント分析（全指数同時表示）=====
    cp = st.session_state.get("clicked_point")
    with st.expander("ポイント分析", expanded=cp is not None):
        if cp is None:
            st.info("地図上をクリックするとポイント値を表示します。")
        else:
            lat, lng = cp["lat"], cp["lng"]
            st.caption(f"緯度: {lat:.5f}, 経度: {lng:.5f}")
            val_cols = st.columns(len(INDEX_CONFIG))
            for col, (index_name, cfg) in zip(val_cols, INDEX_CONFIG.items()):
                data = computed[current_sid].get(index_name)
                if isinstance(data, tuple):
                    val = extract_pixel_value(data[0], data[1], lat, lng)
                    col.metric(cfg["label"], f"{val:.4f}" if not np.isnan(val) else "—")

            # 時系列チャート（複数シーンのみ、3指数を1グラフで比較）
            if len(scene_ids) > 1:
                ts_colors = {"NDVI": "steelblue", "EVI": "seagreen", "NDRE": "coral"}
                fig_ts, axes = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
                has_data = False
                for index_name in INDEX_CONFIG:
                    ts_vals, ts_dates = [], []
                    for sid in scene_ids:
                        d = computed[sid].get(index_name)
                        it = next((i for i in items_all if i.id == sid), None)
                        if isinstance(d, tuple) and it:
                            v = extract_pixel_value(d[0], d[1], lat, lng)
                            ts_vals.append(v)
                            ts_dates.append(
                                date.fromisoformat(it.properties.get("datetime", "")[:10])
                            )
                    if not ts_vals:
                        continue
                    has_data = True
                    color = ts_colors[index_name]
                    axes[0].plot(ts_dates, ts_vals, "o-", color=color, label=index_name)
                    if len(ts_vals) >= 2:
                        day_gaps = [
                            max((ts_dates[i + 1] - ts_dates[i]).days, 1)
                            for i in range(len(ts_dates) - 1)
                        ]
                        rates = [d / g for d, g in zip(np.diff(ts_vals), day_gaps)]
                        mid_dates = [
                            ts_dates[i] + timedelta(days=day_gaps[i] // 2)
                            for i in range(len(ts_dates) - 1)
                        ]
                        axes[1].bar(mid_dates, rates,
                                    width=[g * 0.6 for g in day_gaps],
                                    alpha=0.6, color=color, label=index_name)
                if has_data:
                    axes[0].legend(fontsize=8)
                    axes[0].set_ylabel("指数値")
                    axes[0].set_title("時系列比較")
                    axes[1].axhline(0, color="black", linewidth=0.8)
                    axes[1].set_ylabel("変化率 (Δ/日)")
                    axes[1].legend(fontsize=8)
                    fmt = mdates.DateFormatter("%m/%d")
                    axes[0].xaxis.set_major_formatter(fmt)
                    axes[1].xaxis.set_major_formatter(fmt)
                    fig_ts.autofmt_xdate(rotation=45)
                    fig_ts.tight_layout()
                    st.pyplot(fig_ts)
                plt.close(fig_ts)

    # ===== 統計情報 + PNG ダウンロード タブ =====
    st.subheader("統計情報 / ダウンロード")
    tabs = st.tabs([INDEX_CONFIG[k]["label"] for k in INDEX_CONFIG])
    for tab, index_name in zip(tabs, INDEX_CONFIG):
        with tab:
            data = computed[current_sid].get(index_name)
            if not isinstance(data, tuple):
                st.warning(f"{index_name}: 計算できませんでした。")
                if isinstance(data, str):
                    st.code(data)
                continue
            arr, geo_bounds = data
            cfg = INDEX_CONFIG[index_name]

            valid = arr[~np.isnan(arr)]
            if valid.size:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("平均", f"{valid.mean():.3f}")
                c2.metric("中央値", f"{np.median(valid):.3f}")
                c3.metric("最小", f"{valid.min():.3f}")
                c4.metric("最大", f"{valid.max():.3f}")
            else:
                st.write("有効な画素がありません。")

            scene_date = current_item.properties.get("datetime", "")[:10]
            title = f"{index_name} — {scene_date}"
            fig_dl = render_index_figure(arr, geo_bounds, cfg["vmin"], cfg["vmax"], index_name, title)
            buf = BytesIO()
            fig_dl.savefig(buf, format="png", dpi=150)
            plt.close(fig_dl)
            st.download_button(
                "PNG をダウンロード",
                buf.getvalue(),
                file_name=f"{index_name.lower()}_{current_sid[:10]}.png",
                mime="image/png",
                key=f"dl_{index_name}_{current_sid}",
            )
