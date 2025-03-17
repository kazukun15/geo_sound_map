import os
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import math
import io
import branca.colormap as cm
import requests
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor

# st.set_page_config() は必ず最初に呼び出す
st.set_page_config(page_title="防災スピーカー音圧可視化マップ", layout="wide")

# ---------- Custom CSS for UI styling (UI改善) ----------
custom_css = """
<style>
body { font-family: 'Helvetica', sans-serif; }
h1, h2, h3, h4, h5, h6 { color: #333333; }
div.stButton > button {
    background-color: #4CAF50; 
    color: white; 
    border: none;
    padding: 10px 24px; 
    font-size: 16px; 
    border-radius: 8px; 
    cursor: pointer;
}
div.stButton > button:hover { background-color: #45a049; }
div.stTextInput>div>input { font-size: 16px; padding: 8px; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-weight: bold; color: #4CAF50;
}
/* チャットバブルの統一スタイル */
.chat-bubble {
    background-color: #e8f5e9;
    border-radius: 10px;
    padding: 8px;
    margin: 4px 0;
    max-width: 80%;
    word-wrap: break-word;
}
.chat-header { font-weight: bold; margin-bottom: 4px; color: #4CAF50; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# ---------- End Custom CSS ----------

# ------------------------------------------------------------------
# 定数／設定（APIキー、モデル）
# ------------------------------------------------------------------
API_KEY = st.secrets["general"]["api_key"]  # secrets.toml に [general] セクションで設定
MODEL_NAME = "gemini-2.0-flash"

# ----------------------------------------------------------------
# Module: Direction Utilities
# ----------------------------------------------------------------
DIRECTION_MAPPING = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction(direction_str):
    """
    文字列から方向（度数）に変換する関数。
    例: "N" -> 0, "SW" -> 225, "45" -> 45.0
    """
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    try:
        return float(direction_str)
    except ValueError:
        st.error(f"入力された方向 '{direction_str}' は不正です。0度に設定します。")
        return 0.0

# ----------------------------------------------------------------
# Module: CSV Utilities
# ----------------------------------------------------------------
def load_csv(file):
    """
    CSVファイルを読み込み、スピーカーと計測データを抽出する関数。
    Returns:
        speakers: [[lat, lon, [dir1, dir2, ...]], ...]
        measurements: [[lat, lon, db], ...]
    ※ 不正な行は警告表示してスキップします。
    """
    try:
        df = pd.read_csv(file)
        speakers, measurements = [], []
        for idx, row in df.iterrows():
            try:
                if not pd.isna(row.get("スピーカー緯度")):
                    lat = float(row["スピーカー緯度"])
                    lon = float(row["スピーカー経度"])
                    directions = [parse_direction(row.get(f"方向{i}", "")) 
                                  for i in range(1, 4) if not pd.isna(row.get(f"方向{i}"))]
                    speakers.append([lat, lon, directions])
                if not pd.isna(row.get("計測位置緯度")):
                    lat = float(row["計測位置緯度"])
                    lon = float(row["計測位置経度"])
                    db = float(row.get("計測デシベル", 0))
                    measurements.append([lat, lon, db])
            except Exception as e:
                st.warning(f"行 {idx+1} の読み込みに失敗しました: {e}")
        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込み全体でエラー: {e}")
        return [], []

def export_csv(data, columns):
    """
    スピーカー情報または計測情報をCSV形式にエクスポートする関数。
    """
    rows = []
    for entry in data:
        if len(entry) == 3 and isinstance(entry[2], list):
            lat, lon, directions = entry
            row = {
                "スピーカー緯度": lat,
                "スピーカー経度": lon,
                "方向1": directions[0] if len(directions) > 0 else "",
                "方向2": directions[1] if len(directions) > 1 else "",
                "方向3": directions[2] if len(directions) > 2 else ""
            }
        else:
            row = {columns[i]: entry[i] for i in range(len(columns))}
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ----------------------------------------------------------------
# Module: Heatmap Calculation & Sound Grid Utilities (パフォーマンス最適化)
# ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """ヒートマップデータ計算をキャッシュして、再計算を回避する。"""
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各グリッド点の音圧レベルを計算する関数。
    重い計算処理のため、キャッシュなどを利用して再計算を最小限にします。
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, dirs = spk
        dlat = grid_lat - lat
        dlon = grid_lon - lon
        # 経度は緯度に依存して補正
        distance = np.sqrt((dlat * 111320)**2 + (dlon * 111320 * np.cos(np.radians(lat)))**2)
        distance[distance < 1] = 1
        # 各グリッド点への方位（度）
        bearing = (np.degrees(np.arctan2(dlon, dlat))) % 360
        power = np.zeros_like(distance)
        for direction in dirs:
            angle_diff = np.abs(bearing - direction) % 360
            directional_factor = np.where(np.cos(np.radians(angle_diff)) > 0.3,
                                          np.cos(np.radians(angle_diff)),
                                          0.3)
            intensity = directional_factor * (10 ** ((L0 - 20 * np.log10(distance)) / 10))
            power += intensity
        power[distance > r_max] = 0
        power_sum += power
    sound_grid = np.full_like(power_sum, np.nan)
    valid = power_sum > 0
    sound_grid[valid] = 10 * np.log10(power_sum[valid])
    sound_grid = np.clip(sound_grid, L0 - 40, L0)
    return sound_grid

def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """ヒートマップ用のデータリストを作成する関数。"""
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    heat_data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                heat_data.append([grid_lat[i, j], grid_lon[i, j], val])
    return heat_data

def calculate_objective(speakers, target, L0, r_max, grid_lat, grid_lon):
    """目標音圧レベルとの差の二乗平均誤差（MSE）を計算する関数。"""
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    valid = ~np.isnan(sound_grid)
    mse = np.mean((sound_grid[valid] - target)**2)
    return mse

def optimize_speaker_placement(speakers, target, L0, r_max, grid_lat, grid_lon, iterations=10, delta=0.0001):
    """
    各スピーカーの位置を微調整して、目標音圧レベルとの差（二乗平均誤差）を最小化する自動最適配置アルゴリズム。
    """
    optimized = [list(spk) for spk in speakers]
    current_obj = calculate_objective(optimized, target, L0, r_max, grid_lat, grid_lon)
    
    for _ in range(iterations):
        for i, spk in enumerate(optimized):
            best_spk = spk.copy()
            best_obj = current_obj
            for d_lat in [delta, -delta, 0]:
                for d_lon in [delta, -delta, 0]:
                    if d_lat == 0 and d_lon == 0:
                        continue
                    candidate = spk.copy()
                    candidate[0] += d_lat
                    candidate[1] += d_lon
                    temp_speakers = optimized.copy()
                    temp_speakers[i] = candidate
                    candidate_obj = calculate_objective(temp_speakers, target, L0, r_max, grid_lat, grid_lon)
                    if candidate_obj < best_obj:
                        best_obj = candidate_obj
                        best_spk = candidate.copy()
            optimized[i] = best_spk
            current_obj = calculate_objective(optimized, target, L0, r_max, grid_lat, grid_lon)
    return optimized

# ----------------------------------------------------------------
# Module: Gemini API Utilities & プロンプト生成
# ----------------------------------------------------------------
def generate_gemini_prompt(user_query):
    """
    ユーザーの問い合わせと現在のスピーカー配置、音圧分布情報を元に、改善案を提案するプロンプトを生成します。
    以下の条件を加味してください：
      - 海など設置困難な場所は除外
      - スピーカー同士は300m以上離れること
      - 座標表記は「緯度 xxx.xxxxxx, 経度 yyy.yyyyyy」で統一
    """
    speakers = st.session_state.speakers if "speakers" in st.session_state else []
    num_speakers = len(speakers)
    if num_speakers > 0:
        speaker_info = "配置されているスピーカー:\n" + "\n".join(
            f"{i+1}. 緯度: {s[0]:.6f}, 経度: {s[1]:.6f}, 方向: {s[2]}"
            for i, s in enumerate(speakers)
        )
    else:
        speaker_info = "現在、スピーカーは配置されていません。\n"
    sound_range = f"{st.session_state.L0 - 40}dB ~ {st.session_state.L0}dB"
    prompt = (
        f"{speaker_info}\n"
        f"現在の音圧レベルの範囲: {sound_range}\n"
        "海など設置困難な場所は除外し、スピーカー同士は300m以上離れてい
