import math
import io
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

# ------------------ 定数・設定 ------------------
APP_TITLE    = "上島町 防災無線スピーカ設置場所考えるくん"
MAP_CENTER   = (34.25, 133.20)
DEFAULT_ZOOM = 11
CUSTOM_CSS   = """
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
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for lat_s, lon_s, *_rest, direction in speakers:
        dlat = (grid_lat - lat_s) * 111320
        dlon = (grid_lon - lon_s) * 111320 * math.cos(math.radians(lat_s))
        distance = np.hypot(dlat, dlon)
        distance = np.maximum(distance, 1)
        p_db = L0 - 20 * np.log10(distance)
        bearing = (np.degrees(np.arctan2(dlon, dlat))) % 360
        diff = np.minimum(np.abs(bearing - direction), 360 - np.abs(bearing - direction))
        factor = np.clip(np.cos(np.radians(diff)), 0.3, 1.0)
        power = factor * 10**(p_db/10)
        power[distance > r_max] = 0
        power_sum += power
    return np.where(power_sum>0,
                    np.clip(10*np.log10(power_sum), L0-40, L0),
                    np.nan)

# ------------------ グリッド生成 ------------------
def generate_grid(resolution: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max = 34.20, 34.35
    lon_min, lon_max = 133.15, 133.28
    lats = np.linspace(lat_min, lat_max, resolution)
    lons = np.linspace(lon_min, lon_max, resolution)
    return np.meshgrid(lats, lons, indexing="xy")

# ------------------ 3Dカラム用データ生成 ------------------
@st.cache_data
def get_column_data(
    speakers: List[List], L0: float, r_max: float,
    grid_lat: np.ndarray, grid_lon: np.ndarray
) -> pd.DataFrame:
    sgrid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    mask = ~np.isnan(sgrid)
    if not mask.any():
        return pd.DataFrame()
    vmin, vmax = np.nanmin(sgrid), np.nanmax(sgrid)
    data = []
    for (i,j) in np.argwhere(mask):
        val = sgrid[i,j]
        norm = (val - vmin)/(vmax-vmin) if vmax>vmin else 0
        elevation = norm*500
        r = int(255*norm); g = int(255*(1-norm)); b = 128; a = int(120*norm)
        data.append({"lon":grid_lon[i,j],"lat":grid_lat[i,j],
                     "elevation":elevation,"color":[r,g,b,a]})
    return pd.DataFrame(data)

# ------------------ レイヤー作成 ------------------
def create_heatmap_layer(df: pd.DataFrame, zoom: int) -> pdk.Layer:
    def dynamic_radius(z): return max(200, int(50*(30/z)))
    return pdk.Layer(
        "HeatmapLayer", df,
        get_position=["longitude","latitude"],
        get_weight="weight",
        radiusPixels=dynamic_radius(zoom),
        opacity=0.3,
        threshold=0.05
    )

def create_column_layer(df: pd.DataFrame) -> pdk.Layer:
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
    return pdk.Layer(
        "ScatterplotLayer", df,
        get_position=["lon","lat"],
        get_fill_color=[0,0,255,255],
        get_radius=30,
        pickable=True, auto_highlight=True
    )

def create_scenegraph_layer(df: pd.DataFrame) -> pdk.Layer:
    URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/scenegraph/airplane/scene.gltf"
    return pdk.Layer(
        "ScenegraphLayer", df.assign(z=50),
        scenegraph=URL,
        get_position=["lon","lat","z"],
        sizeScale=20,
        pickable=True
    )

# ------------------ Gemini連携 ------------------
def generate_gemini_prompt(user_query: str, speakers: List[List], L0: float) -> str:
    spk_info = "\n".join(
        f"{i+1}. 緯度:{s[0]:.6f}, 経度:{s[1]:.6f}, ラベル:{s[2]}, 方向:{s[3]}"
        for i,s in enumerate(speakers)
    ) or "スピーカー未配置"
    return (
        f"現在のスピーカー配置:\n{spk_info}\n"
        f"初期音圧レベル: {L0}dB\n"
        "地形情報（海域、山岳、道路網など）を考慮し、スピーカー同士は300m以上離しつつ"
        "未カバー領域にも注意し、多角的に最適な設置場所を提案してください。\n"
        f"ユーザー要求: {user_query}\n"
        "【回答フォーマット】緯度:xx.xxxxxx, 経度:yy.yyyyyy, 方向:ZZ\n"
    )

def call_gemini(prompt: str, api_key: str, model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents":[{"parts":[{"text":prompt}]}]}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        cands = r.json().get("candidates",[])
        if not cands: return ""
        cv = cands[0].get("content",cands[0].get("output",""))
        if isinstance(cv,dict):
            return "".join(p.get("text","") for p in cv.get("parts",[]))
        return str(cv)
    except Exception as e:
        st.error(f"Gemini APIエラー: {e}")
        return ""

def extract_speaker_proposals(text: str) -> List[List]:
    pattern = r"緯度[:：]\s*([-\d\.]+)[,、]\s*経度[:：]\s*([-\d\.]+)(?:[,、]\s*方向[:：]?\s*(\d+))?"
    props = re.findall(pattern, text)
    results = []
    for lat_s, lon_s, dir_s in props:
        try:
            lat = float(lat_s); lon = float(lon_s)
            direction = parse_direction(dir_s or "0")
            results.append([lat, lon, "Gemini提案", direction])
        except:
            continue
    return results

# ------------------ Streamlitレイアウト ------------------
def setup_page():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title(APP_TITLE)

def init_state():
    s = st.session_state
    s.setdefault("speakers", [[34.25,133.20,"初期スピーカーA",0.0]])
    s.setdefault("L0",80)
    s.setdefault("r_max",500)
    s.setdefault("map_zoom",DEFAULT_ZOOM)
    s.setdefault("grid", generate_grid())
    s.setdefault("heatmap_df",None)
    s.setdefault("display_mode","HeatMap")
    s.setdefault("center_targets",[])
    s.setdefault("gemini_query","")
    s.setdefault("gemini_result","")

def main():
    setup_page()
    init_state()

    # サイドバー
    with st.sidebar:
        up = st.file_uploader("CSVアップロード", type="csv")
        if up and st.button("登録"):
            new = load_speakers_from_csv(up)
            st.session_state.speakers.extend(new)
            st.session_state.heatmap_df = None
            st.success(f"{len(new)}件追加")

        txt = st.text_input("追加 (lat,lon,label,方向)")
        if st.button("追加"):
            try:
                lat, lon, *rest = [p.strip() for p
