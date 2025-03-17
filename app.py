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
      - スピーカー同士は300m以上離れていること
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
        "海など設置困難な場所は除外し、スピーカー同士は300m以上離れている場所を考慮してください。\n"
        f"ユーザーの問い合わせ: {user_query}\n"
        "上記情報に基づき、改善案を具体的かつ詳細に提案してください。\n"
        "【座標表記形式】 緯度 xxx.xxxxxx, 経度 yyy.yyyyyy で統一してください。"
    )
    return prompt

def call_gemini_api(query):
    """Gemini API にクエリを送信する関数。"""
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload = {"contents": [{"parts": [{"text": query}]}]}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        rjson = response.json()
        candidates = rjson.get("candidates", [])
        if not candidates:
            st.error("Gemini API エラー: candidatesが空")
            return "回答が得られませんでした。"
        candidate0 = candidates[0]
        content_val = candidate0.get("content", "")
        if isinstance(content_val, dict):
            parts = content_val.get("parts", [])
            content_str = " ".join([p.get("text", "") for p in parts])
        else:
            content_str = str(content_val)
        return content_str.strip()
    except Exception as e:
        st.error(f"Gemini API呼び出しエラー: {e}")
        return f"エラー: {e}"

# ----------------------------------------------------------------
# ThreadPoolExecutor（非同期処理）
# ----------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=2)

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """ヒートマップデータ計算をキャッシュして、再計算を回避する。"""
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

# -----------------------------------------------------------------------------
# Main Application (UI) – メインパネル・UI改善
# -----------------------------------------------------------------------------
def main():
    st.title("防災スピーカー音圧可視化マップ")
    
    # セッション初期化
    if "map_center" not in st.session_state:
        st.session_state.map_center = [34.25741795269067, 133.20450105700033]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 14
    if "speakers" not in st.session_state:
        st.session_state.speakers = [[34.25741795269067, 133.20450105700033, [0.0, 90.0]]]
    if "heatmap_data" not in st.session_state:
        st.session_state.heatmap_data = None
    if "L0" not in st.session_state:
        st.session_state.L0 = 80
    if "r_max" not in st.session_state:
        st.session_state.r_max = 500
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gemini_result" not in st.session_state:
        st.session_state.gemini_result = None
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = None
    
    with st.sidebar:
        st.header("操作パネル")
        
        # CSVファイルアップロードと「CSVからスピーカー登録」ボタン
        uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        if uploaded_file:
            if st.button("CSVからスピーカー登録"):
                speakers, _ = load_csv(uploaded_file)
                if speakers:
                    st.session_state.speakers.extend(speakers)
                    st.success("CSVからスピーカー情報を登録しました。")
                else:
                    st.error("CSVに正しいデータが見つかりませんでした。")
                st.session_state.heatmap_data = None
        
        # スピーカー手動追加（入力チェックあり）
        new_speaker = st.text_input("スピーカー追加 (緯度,経度,方向1,方向2,...)",
                                    placeholder="例: 34.2579,133.2072,N,E")
        if st.button("スピーカー追加"):
            parts = new_speaker.split(",")
            if len(parts) < 3:
                st.error("入力形式が正しくありません。少なくとも緯度, 経度, 方向1が必要です。")
            else:
                try:
                    lat, lon = float(parts[0]), float(parts[1])
                    directions = [parse_direction(x) for x in parts[2:]]
                    st.session_state.speakers.append([lat, lon, directions])
                    st.session_state.heatmap_data = None
                    st.success(f"スピーカー追加成功: 緯度 {lat}, 経度 {lon}, 方向 {directions}")
                except Exception as e:
                    st.error(f"スピーカー追加エラー: {e}")
        
        # スピーカー削除・編集
        if st.session_state.speakers:
            options = [f"{i}: ({s[0]:.6f}, {s[1]:.6f}) - 方向: {s[2]}" for i, s in enumerate(st.session_state.speakers)]
            selected = st.selectbox("スピーカーを選択", list(range(len(options))), format_func=lambda i: options[i])
            col_del, col_edit = st.columns(2)
            with col_del:
                if st.button("選択したスピーカーを削除"):
                    try:
                        del st.session_state.speakers[selected]
                        st.session_state.heatmap_data = None
                        st.success("スピーカー削除成功")
                    except Exception as e:
                        st.error(f"削除エラー: {e}")
            with col_edit:
                if st.button("選択したスピーカーを編集"):
                    st.session_state.edit_index = selected
        else:
            st.info("スピーカーがありません。")
        
        if st.session_state.get("edit_index") is not None:
            with st.form("edit_form"):
                spk = st.session_state.speakers[st.session_state.edit_index]
                new_lat = st.text_input("新しい緯度", value=str(spk[0]))
                new_lon = st.text_input("新しい経度", value=str(spk[1]))
                new_dirs = st.text_input("新しい方向（カンマ区切り）", value=",".join(str(x) for x in spk[2]))
                submitted = st.form_submit_button("編集保存")
                if submitted:
                    try:
                        lat_val = float(new_lat)
                        lon_val = float(new_lon)
                        directions_val = [parse_direction(x) for x in new_dirs.split(",")]
                        st.session_state.speakers[st.session_state.edit_index] = [lat_val, lon_val, directions_val]
                        st.session_state.heatmap_data = None
                        st.success("スピーカー情報更新成功")
                        st.session_state.edit_index = None
                    except Exception as e:
                        st.error(f"編集保存エラー: {e}")
        
        if st.button("スピーカーリセット"):
            st.session_state.speakers = []
            st.session_state.heatmap_data = None
            st.success("スピーカーリセット完了")
        
        st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
        
        target_default = st.session_state.L0 - 20
        target_level = st.slider("目標音圧レベル (dB)", st.session_state.L0 - 40, st.session_state.L0, target_default)
        if st.button("自動最適配置を実行"):
            lat_min = st.session_state.map_center[0] - 0.01
            lat_max = st.session_state.map_center[0] + 0.01
            lon_min = st.session_state.map_center[1] - 0.01
            lon_max = st.session_state.map_center[1] + 0.01
            grid_lat, grid_lon = np.meshgrid(
                np.linspace(lat_min, lat_max, 50),
                np.linspace(lon_min, lon_max, 50)
            )
            try:
                optimized = optimize_speaker_placement(
                    st.session_state.speakers,
                    target_level,
                    st.session_state.L0,
                    st.session_state.r_max,
                    grid_lat,
                    grid_lon,
                    iterations=10,
                    delta=0.0001
                )
                st.session_state.speakers = optimized
                st.session_state.heatmap_data = None
                st.success("自動最適配置成功")
            except Exception as e:
                st.error(f"自動最適配置エラー: {e}")
        
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_input("Gemini に問い合わせる内容")
        if st.button("Gemini API を実行"):
            full_prompt = generate_gemini_prompt(gemini_query)
            result = call_gemini_api(full_prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API 実行完了")
    
    # メインパネル：地図とヒートマップの表示
    col1, col2 = st.columns([3, 1])
    with col1:
        lat_min = st.session_state.map_center[0] - 0.01
        lat_max = st.session_state.map_center[0] + 0.01
        lon_min = st.session_state.map_center[1] - 0.01
        lon_max = st.session_state.map_center[1] + 0.01
        grid_lat, grid_lon = np.meshgrid(
            np.linspace(lat_min, lat_max, 100),
            np.linspace(lon_min, lon_max, 100)
        )
        if st.session_state.heatmap_data is None:
            st.session_state.heatmap_data = cached_calculate_heatmap(
                st.session_state.speakers,
                st.session_state.L0,
                st.session_state.r_max,
                grid_lat,
                grid_lon
            )
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
        for spk in st.session_state.speakers:
            lat, lon, dirs = spk
            popup_text = f"<b>スピーカー</b>: ({lat:.6f}, {lon:.6f})<br><b>方向</b>: {dirs}"
            folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300)).add_to(m)
        if st.session_state.heatmap_data:
            HeatMap(st.session_state.heatmap_data, min_opacity=0.3, max_opacity=0.8, radius=15, blur=20).add_to(m)
        st_folium(m, width=700, height=500)
    with col2:
        csv_data = export_csv(
            st.session_state.speakers,
            ["スピーカー緯度", "スピーカー経度", "方向1", "方向2", "方向3"]
        )
        st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        count = len(st.session_state.heatmap_data) if st.session_state.heatmap_data else 0
        st.write("ヒートマップデータの件数:", count)
    
    st.markdown("---")
    st.subheader("Gemini API の回答（JSON含む）")
    if st.session_state.gemini_result:
        st.json(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答はまだありません。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
