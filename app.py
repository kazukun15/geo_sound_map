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
/* ポップアップやチャット風のバブルを少し整える */
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
# CSV読み取りの修正ポイント
#  - "latitude", "longitude" カラムを読み取る
#  - "施設名" または "名称" カラムがあれば、それをラベルとして読み取る
# ----------------------------------------------------------------

def load_csv(file):
    """
    CSVファイルを読み込み、スピーカーの位置情報を抽出する関数。
    - "latitude", "longitude" を必須カラムとし、
      もし "施設名" または "名称" カラムがあれば、その値をラベルとして格納。
    - 読み取ったデータは [[lat, lon, label], ...] の形式で返す。
    """
    try:
        df = pd.read_csv(file)
        speakers = []
        for idx, row in df.iterrows():
            try:
                # "latitude" "longitude" カラムがあるかチェック
                if not pd.isna(row.get("latitude")) and not pd.isna(row.get("longitude")):
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                    # 施設名や名称などがあれば label として読み込む
                    label = ""
                    if "施設名" in df.columns and not pd.isna(row.get("施設名")):
                        label = str(row["施設名"]).strip()
                    elif "名称" in df.columns and not pd.isna(row.get("名称")):
                        label = str(row["名称"]).strip()
                    
                    # speakers リストに [lat, lon, label] で格納
                    speakers.append([lat, lon, label])
                else:
                    # latitude, longitude のどちらかが欠損している場合はスキップ
                    st.warning(f"行 {idx+1} に 'latitude' または 'longitude' がありません。スキップします。")
            except Exception as e:
                st.warning(f"行 {idx+1} の読み込みに失敗しました: {e}")
        return speakers
    except Exception as e:
        st.error(f"CSV読み込み全体でエラー: {e}")
        return []

def export_csv(data):
    """
    スピーカー情報をCSV形式にエクスポートする関数。
    data: [[lat, lon, label], ...]
    CSVの列名は ["latitude", "longitude", "label"] とする。
    """
    rows = []
    for entry in data:
        if len(entry) == 3:
            lat, lon, label = entry
            row = {
                "latitude": lat,
                "longitude": lon,
                "label": label
            }
            rows.append(row)
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "label"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ----------------------------------------------------------------
# ヒートマップ計算の関数（最小限の実装）
# ----------------------------------------------------------------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    シンプルな音圧レベル計算の例:
      - speakers: [[lat, lon, label], ...]
      - L0: 初期音圧レベル
      - r_max: 最大伝播距離
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, label = spk
        dlat = grid_lat - lat
        dlon = grid_lon - lon
        distance = np.sqrt((dlat * 111320)**2 + (dlon * 111320 * np.cos(np.radians(lat)))**2)
        distance[distance < 1] = 1
        power = L0 - 20 * np.log10(distance)
        power[distance > r_max] = -999
        # ここでは方向やlabelは使用せず、単純に足し合わせる例
        # 必要に応じて加重や方向性を加味してください
        valid_idx = (power > -999)
        power_sum[valid_idx] += 10 ** (power[valid_idx] / 10)
    sound_grid = np.full_like(power_sum, np.nan)
    valid = power_sum > 0
    sound_grid[valid] = 10 * np.log10(power_sum[valid])
    return sound_grid

@st.cache_data(show_spinner=False)
def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    heat_data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                heat_data.append([grid_lat[i, j], grid_lon[i, j], val])
    return heat_data

# ----------------------------------------------------------------
# メインアプリ
# ----------------------------------------------------------------
def main():
    st.title("防災スピーカー音圧可視化マップ（latitude, longitude, label 対応版）")
    
    if "map_center" not in st.session_state:
        st.session_state.map_center = [34.25741795269067, 133.20450105700033]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 14
    if "speakers" not in st.session_state:
        # 初期値
        st.session_state.speakers = [
            [34.25741795269067, 133.20450105700033, "初期スピーカー1"]
        ]
    if "heatmap_data" not in st.session_state:
        st.session_state.heatmap_data = None
    if "L0" not in st.session_state:
        st.session_state.L0 = 80
    if "r_max" not in st.session_state:
        st.session_state.r_max = 500
    
    # サイドバー
    with st.sidebar:
        st.header("操作パネル")
        
        uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        if uploaded_file:
            # ボタンを押したらCSVを読み取り、speakers に追加
            if st.button("CSVからスピーカー登録"):
                new_speakers = load_csv(uploaded_file)
                if new_speakers:
                    st.session_state.speakers.extend(new_speakers)
                    st.session_state.heatmap_data = None
                    st.success(f"CSVから {len(new_speakers)} 件のスピーカー情報を追加しました。")
                else:
                    st.error("CSVに正しいデータが見つかりませんでした。")
        
        # スピーカー追加： (latitude, longitude, label) 形式
        new_speaker = st.text_input("スピーカー追加 (latitude,longitude,label)", 
                                    placeholder="例: 34.2579,133.2072,役場")
        if st.button("スピーカー追加"):
            parts = new_speaker.split(",")
            if len(parts) < 2:
                st.error("入力形式が正しくありません。(latitude,longitude,label)")
            else:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    label = parts[2].strip() if len(parts) > 2 else ""
                    st.session_state.speakers.append([lat, lon, label])
                    st.session_state.heatmap_data = None
                    st.success(f"スピーカー追加成功: 緯度 {lat}, 経度 {lon}, ラベル {label}")
                except Exception as e:
                    st.error(f"スピーカー追加エラー: {e}")
        
        if st.button("スピーカーリセット"):
            st.session_state.speakers = []
            st.session_state.heatmap_data = None
            st.success("スピーカーをリセットしました")
        
        st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
    
    # メインパネル
    lat_min = st.session_state.map_center[0] - 0.01
    lat_max = st.session_state.map_center[0] + 0.01
    lon_min = st.session_state.map_center[1] - 0.01
    lon_max = st.session_state.map_center[1] + 0.01
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, 100),
        np.linspace(lon_min, lon_max, 100)
    )
    
    if st.session_state.heatmap_data is None:
        st.session_state.heatmap_data = calculate_heatmap(
            st.session_state.speakers,
            st.session_state.L0,
            st.session_state.r_max,
            grid_lat,
            grid_lon
        )
    
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
    for spk in st.session_state.speakers:
        lat, lon, label = spk
        popup_text = f"<b>スピーカー</b>: ({lat:.6f}, {lon:.6f})"
        if label:
            popup_text += f"<br><b>ラベル</b>: {label}"
        folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300)).add_to(m)
    
    if st.session_state.heatmap_data:
        HeatMap(st.session_state.heatmap_data, min_opacity=0.3, max_opacity=0.8, radius=15, blur=20).add_to(m)
    st_folium(m, width=700, height=500)
    
    # CSVダウンロード
    csv_data = export_csv(st.session_state.speakers)
    st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        count = len(st.session_state.heatmap_data) if st.session_state.heatmap_data else 0
        st.write("ヒートマップデータの件数:", count)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
