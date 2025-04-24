import os
import math
import io
import re
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

# ------------------ 定数・設定 ------------------
APP_TITLE = "上島町 防災無線スピーカ設置場所考えるくん"
MAP_CENTER = (34.25, 133.20)
DEFAULT_ZOOM = 11
CUSTOM_CSS = """
<style>
body { font-family: 'Helvetica', sans-serif; }
h1, h2, h3, h4, h5, h6 { color: #333333; }
div.stButton > button {
    background-color: #4CAF50; color: white; border: none;
    padding: 10px 24px; font-size: 16px; border-radius: 8px; cursor: pointer; text-align: left;
}
div.stButton > button:hover { background-color: #45a049; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-weight: bold; color: #4CAF50;
}
</style>
"""
DIRECTION_MAP = {"N":0, "NE":45, "E":90, "SE":135, "S":180, "SW":225, "W":270, "NW":315}

# ------------------ ユーティリティ ------------------
def parse_direction(dir_str: str) -> float:
    """方位記号または数値文字列を度数に変換"""
    s = dir_str.strip().upper()
    if s in DIRECTION_MAP:
        return float(DIRECTION_MAP[s])
    try:
        return float(s)
    except ValueError:
        st.warning(f"方向 '{dir_str}' は不正です。0° とみなします。")
        return 0.0

# ------------------ CSV I/O ------------------
def load_speakers_from_csv(file) -> List[List]:
    df = pd.read_csv(file)
    speakers = []
    for idx, row in df.iterrows():
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            label = row.get("label") or row.get("施設名") or row.get("名称") or ""
            direction = parse_direction(str(row.get("direction", "0")))
            speakers.append([lat, lon, label.strip(), direction])
        except Exception as e:
            st.warning(f"行 {idx+1} 読み込み失敗: {e}")
    return speakers

def speakers_to_csv(speakers: List[List]) -> bytes:
    rows = [{"latitude": lat, "longitude": lon, "label": label}
            for lat, lon, label, *_ in speakers]
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ------------------ 音圧計算 ------------------
@st.cache_data
def compute_sound_grid(
    speakers: List[List], L0: float, r_max: float,
    grid_lat: np.ndarray, grid_lon: np.ndarray
) -> np.ndarray:
    """
    音源ごとの音圧を合成し、dBスケールで返す。
    方向補正と距離減衰を考慮。
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for lat_s, lon_s, *_rest, direction in speakers:
        dlat = (grid_lat - lat_s) * 111320
        dlon = (grid_lon - lon_s) * 111320 * math.cos(math.radians(lat_s))
        distance = np.hypot(dlat, dlon)
        distance = np.maximum(distance, 1)
        p_db = L0 - 20 * np.log10(distance)
        bearing = (np.degrees(np.arctan2(dlon, dlat))) % 360
        diff = np.abs(bearing - direction)
        diff = np.minimum(diff, 360 - diff)
        factor = np.clip(np.cos(np.radians(diff)), 0.3, 1.0)
        power = factor * 10**(p_db/10)
        power[distance > r_max] = 0
        power_sum += power

    total_db = np.where(
        power_sum>0,
        np.clip(10*np.log10(power_sum), L0-40, L0),
        np.nan
    )
    return total_db

# ------------------ グリッド生成 ------------------
def generate_grid(resolution: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max = 34.20, 34.35
    lon_min, lon_max = 133.15, 133.28
    lats = np.linspace(lat_min, lat_max, resolution)
    lons = np.linspace(lon_min, lon_max, resolution)
    return np.meshgrid(lats, lons, indexing="xy")

# ------------------ レイヤー作成 ------------------
def create_heatmap_layer(df: pd.DataFrame, zoom: int) -> pdk.Layer:
    """HeatmapLayerを返す"""
    def dynamic_radius(z): return max(200, int(50*(30/z)))
    return pdk.Layer(
        "HeatmapLayer", df,
        get_position=["longitude","latitude"],
        get_weight="weight",
        radiusPixels=dynamic_radius(zoom),
        min_opacity=0.1, max_opacity=0.3
    )

def create_column_layer(df: pd.DataFrame) -> pdk.Layer:
    """ColumnLayerを返す"""
    return pdk.Layer(
        "ColumnLayer", df,
        get_position=["lon","lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=20,
        get_fill_color="color",
        pickable=True, auto_highlight=True
    )

def create_scatter_layer(df: pd.DataFrame) -> pdk.Layer:
    """追加スピーカー用ScatterplotLayer"""
    return pdk.Layer(
        "ScatterplotLayer", df,
        get_position=["lon","lat"],
        get_fill_color=[0,0,255,255],
        get_radius=30,
        pickable=True, auto_highlight=True
    )

def create_scenegraph_layer(df: pd.DataFrame) -> pdk.Layer:
    """初期スピーカーの3DモデルLayer"""
    URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/scenegraph/airplane/scene.gltf"
    return pdk.Layer(
        "ScenegraphLayer", df.assign(z=50),
        scenegraph=URL,
        get_position=["lon","lat","z"],
        sizeScale=20,
        pickable=True
    )

# ------------------ Gemini API連携 ------------------
def call_gemini(prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents":[{"parts":[{"text":prompt}]}]}
    try:
        res = requests.post(url, json=payload, timeout=30)
        res.raise_for_status()
        cands = res.json().get("candidates", [])
        text = ""
        if cands:
            cv = cands[0].get("content", cands[0].get("output",""))
            text = "".join(p.get("text","") for p in (cv.get("parts",[]) if isinstance(cv,dict) else []))
        return text or "回答なし"
    except Exception as e:
        st.error(f"Gemini APIエラー: {e}")
        return ""

# ------------------ Streamlitレイアウト ------------------
def setup_page():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title(APP_TITLE)

def init_state():
    state = st.session_state
    state.setdefault("speakers", [[34.25,133.20,"初期スピーカーA",0.0]])
    state.setdefault("L0", 80)
    state.setdefault("r_max", 500)
    state.setdefault("map_zoom", DEFAULT_ZOOM)
    state.setdefault("grid", generate_grid())
    state.setdefault("heatmap_df", None)
    state.setdefault("gemini_res", "")

def main():
    setup_page()
    init_state()

    # --- サイドバー操作 ---
    with st.sidebar:
        up = st.file_uploader("CSVアップロード", type="csv")
        if up and st.button("登録"):
            new = load_speakers_from_csv(up)
            st.session_state.speakers.extend(new)
            st.session_state.heatmap_df = None
            st.success(f"{len(new)}件追加")

        txt = st.text_input("追加 (lat,lon,label,方向)", "")
        if st.button("追加"):
            try:
                lat, lon, *rest = [p.strip() for p in txt.split(",")]
                lbl = rest[0] if rest else "新スピーカー"
                dir_deg = parse_direction(rest[1] if len(rest)>1 else "0")
                st.session_state.speakers.append([float(lat),float(lon),lbl,dir_deg])
                st.session_state.heatmap_df = None
                st.success("追加成功")
            except Exception as e:
                st.error(f"追加失敗: {e}")

        st.slider("初期音圧 (dB)", 50,100, key="L0")
        st.slider("最大伝播距離 (m)",100,2000, key="r_max")
        mode = st.radio("表示モード", ["HeatMap","3D Columns"], index=0)
        st.session_state.display_mode = mode

        query = st.text_input("Gemini 問い合わせ")
        if st.button("Gemini 実行"):
            prompt = query  # 必要に応じてgenerate_promptで整形
            res = call_gemini(prompt, st.secrets["general"]["api_key"], "gemini-2.0-flash")
            st.session_state.gemini_res = res
            st.success("Gemini 完了")

    # --- グリッド取得 ---
    grid_lat, grid_lon = st.session_state.grid

    # --- ヒートマップ/カラム生成 ---
    if st.session_state.heatmap_df is None:
        sgrid = compute_sound_grid(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat, grid_lon
        )
        data = [
            {"latitude":grid_lat[i,j],"longitude":grid_lon[i,j],"weight":sgrid[i,j]}
            for i,j in np.ndindex(sgrid.shape)
        ]
        st.session_state.heatmap_df = pd.DataFrame(data)

    layers = []
    if st.session_state.display_mode == "HeatMap":
        layers.append(create_heatmap_layer(st.session_state.heatmap_df, st.session_state.map_zoom))
    else:
        # カラム用DF作成省略（compute_sound_grid→正規化→色・elevation計算）
        col_df = ...  # 前述のget_column_dataを参考にデータフレーム作成
        layers.append(create_column_layer(col_df))

    # --- スピーカー表示 ---
    df_init = pd.DataFrame(st.session_state.speakers[:1], columns=["lat","lon","label","direction"])
    df_add  = pd.DataFrame(st.session_state.speakers[1:], columns=["lat","lon","label","direction"])
    layers.append(create_scenegraph_layer(df_init) if not df_init.empty else [])
    layers.append(create_scatter_layer(df_add)    if not df_add.empty else [])

    # --- マップ描画 ---
    view = pdk.ViewState(latitude=*MAP_CENTER, longitude=*MAP_CENTER,
                        zoom=st.session_state.map_zoom,
                        pitch=45 if mode=="3D Columns" else 0)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view))

    # --- CSVダウンロード & デバッグ ---
    st.download_button("CSVダウンロード",
                       speakers_to_csv(st.session_state.speakers),
                       file_name="speakers.csv")
    with st.expander("デバッグ"):
        st.json({
            "speakers": st.session_state.speakers,
            "mode": st.session_state.display_mode,
            "L0": st.session_state.L0,
            "r_max": st.session_state.r_max,
            "gemini": st.session_state.gemini_res
        })

if __name__ == "__main__":
    main()
