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

# ------------------------------------------------------------------
# 定数／設定（APIキー、モデル）
# ------------------------------------------------------------------
API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash"

# ----------------------------------------------------------------
# Module: Direction Utilities
# ----------------------------------------------------------------
DIRECTION_MAPPING = {"N": 0, "E": 90, "S": 180, "W": 270, "NE": 45, "SE": 135, "SW": 225, "NW": 315}

def parse_direction(direction_str):
    """
    文字列から方向（度数）に変換する関数。
    
    Parameters:
        direction_str (str): 入力の方向文字列。例: "N", "45", "SW"
    
    Returns:
        float: 変換された角度（度数）
    """
    direction_str = direction_str.strip().upper()
    if direction_str in DIRECTION_MAPPING:
        return DIRECTION_MAPPING[direction_str]
    try:
        return float(direction_str)
    except ValueError:
        st.error(f"方向 '{direction_str}' を変換できません。0度に設定します。")
        return 0.0

# ----------------------------------------------------------------
# Module: CSV Utilities
# ----------------------------------------------------------------
def load_csv(file):
    """
    CSVファイルを読み込み、スピーカーと計測データを抽出する関数。
    
    Parameters:
        file: アップロードされたCSVファイル
    
    Returns:
        tuple: (speakers, measurements)
            speakers: [ [lat, lon, [direction1, direction2, ...]], ... ]
            measurements: [ [lat, lon, decibel], ... ]
    """
    try:
        df = pd.read_csv(file)
        speakers, measurements = [], []
        for _, row in df.iterrows():
            # スピーカーデータの抽出
            if not pd.isna(row.get("スピーカー緯度")):
                lat, lon = row["スピーカー緯度"], row["スピーカー経度"]
                directions = [parse_direction(row.get(f"方向{i}", "")) for i in range(1, 4) if not pd.isna(row.get(f"方向{i}"))]
                speakers.append([lat, lon, directions])
            # 計測データの抽出
            if not pd.isna(row.get("計測位置緯度")):
                lat, lon, db = row["計測位置緯度"], row["計測位置経度"], row.get("計測デシベル", 0)
                measurements.append([lat, lon, float(db)])
        return speakers, measurements
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return [], []

def export_csv(data, columns):
    """
    スピーカー情報または計測情報をCSV形式にエクスポートする関数。
    
    Parameters:
        data: エクスポートするデータリスト
        columns: CSVのカラム名リスト
    
    Returns:
        bytes: CSVファイルのバイナリデータ
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
# Module: Heatmap Calculation & Sound Grid Utilities
# ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各グリッド点における音圧レベルを計算し、2D配列（sound_grid）として返す関数。
    
    Parameters:
        speakers (list): スピーカーリスト（各要素は [lat, lon, [directions]]）
        L0 (float): 初期音圧レベル (dB)
        r_max (float): 最大伝播距離 (m)
        grid_lat (ndarray): 緯度のグリッド（2D配列）
        grid_lon (ndarray): 経度のグリッド（2D配列）
    
    Returns:
        ndarray: 各グリッド点における音圧レベル（dB）の2D配列
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

    for spk in speakers:
        lat, lon, dirs = spk
        spk_coords = np.array([lat, lon])
        # 距離計算（単位: メートル）
        distances = np.sqrt(np.sum((grid_coords - spk_coords) ** 2, axis=1)) * 111320
        distances[distances < 1] = 1  # 最小距離1mの補正
        bearings = np.degrees(np.arctan2(grid_coords[:, 1] - lon, grid_coords[:, 0] - lat)) % 360
        power = np.zeros_like(distances)
        for direction in dirs:
            angle_diff = np.abs(bearings - direction) % 360
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            power += np.clip(1 - angle_diff / 180, 0, 1) * 10 ** ((L0 - 20 * np.log10(distances)) / 10)
        power[distances > r_max] = 0
        power_sum += power.reshape(Nx, Ny)
    
    sound_grid = np.full_like(power_sum, np.nan)
    positive_mask = power_sum > 0
    sound_grid[positive_mask] = 10 * np.log10(power_sum[positive_mask])
    sound_grid = np.clip(sound_grid, L0 - 40, L0)
    return sound_grid

@st.cache_data(show_spinner=False)
def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """
    ヒートマップ用のデータリストを作成する関数。
    
    Parameters:
        speakers (list): スピーカーリスト
        L0 (float): 初期音圧レベル (dB)
        r_max (float): 最大伝播距離 (m)
        grid_lat, grid_lon (ndarray): 緯度・経度のグリッド
    
    Returns:
        list: ヒートマップデータのリスト [ [lat, lon, sound_level], ... ]
    """
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    heat_data = [[grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]] 
                 for i in range(Nx) for j in range(Ny) if not np.isnan(sound_grid[i, j])]
    return heat_data

def calculate_objective(speakers, target, L0, r_max, grid_lat, grid_lon):
    """
    目標音圧レベルとの差の二乗平均誤差（MSE）を計算する関数。
    
    Parameters:
        speakers (list): スピーカーリスト
        target (float): 目標とする音圧レベル (dB)
        L0 (float): 初期音圧レベル (dB)
        r_max (float): 最大伝播距離 (m)
        grid_lat, grid_lon (ndarray): 緯度・経度のグリッド
    
    Returns:
        float: MSEの値
    """
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    valid = ~np.isnan(sound_grid)
    mse = np.mean((sound_grid[valid] - target) ** 2)
    return mse

def optimize_speaker_placement(speakers, target, L0, r_max, grid_lat, grid_lon, iterations=10, delta=0.0001):
    """
    簡易なヒューリスティック法を用いて、各スピーカーの位置を微調整し、
    目標音圧レベルとの差（二乗誤差）を最小化する自動最適配置アルゴリズム。
    
    Parameters:
        speakers (list): 現在のスピーカーリスト（各要素は [lat, lon, directions]）
        target (float): 目標音圧レベル (dB)
        L0 (float): 初期音圧レベル (dB)
        r_max (float): 最大伝播距離 (m)
        grid_lat, grid_lon (ndarray): 緯度・経度のグリッド
        iterations (int): 最適化の試行回数
        delta (float): 座標変更の刻み幅（緯度・経度の単位）
    
    Returns:
        list: 最適化後のスピーカーリスト
    """
    # 現在のスピーカー配置のコピー
    optimized = [list(spk) for spk in speakers]
    current_obj = calculate_objective(optimized, target, L0, r_max, grid_lat, grid_lon)
    
    for _ in range(iterations):
        for i, spk in enumerate(optimized):
            best_spk = spk.copy()
            best_obj = current_obj
            # 候補：緯度と経度それぞれに対し、+delta, -deltaの移動を試す
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
    ユーザーの問い合わせと現在の地図上のスピーカー配置情報・音圧分布の概要を組み合わせたプロンプトを生成する。
    
    Parameters:
        user_query (str): ユーザーが入力した問い合わせ内容
    
    Returns:
        str: Gemini API へ送信するプロンプト文字列
    """
    speaker_info = ""
    if st.session_state.speakers:
        speaker_info = "現在の地図には以下のスピーカーが配置されています:\n"
        for idx, spk in enumerate(st.session_state.speakers):
            speaker_info += f"{idx+1}. 緯度: {spk[0]:.6f}, 経度: {spk[1]:.6f}, 方向: {spk[2]}\n"
    else:
        speaker_info = "現在、スピーカーは配置されていません。\n"
    
    sound_range = f"{st.session_state.L0-40}dB ~ {st.session_state.L0}dB"
    prompt = (
        f"{speaker_info}"
        f"ヒートマップ解析によると、音圧レベルは概ね {sound_range} の範囲です。\n"
        f"ユーザーからの問い合わせ: {user_query}\n"
        "上記の情報を元に、地図上のスピーカー配置や音圧分布に関する分析、改善案や提案を具体的に述べてください。"
    )
    return prompt

def call_gemini_api(query):
    """
    Gemini APIに対してクエリを送信する関数。
    使用するAPIキーとモデルは streamlit の secrets から取得した定数を利用する。
    
    Parameters:
        query (str): APIに送るクエリ文字列
    
    Returns:
        dict: APIからのレスポンス
    """
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": query}]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Gemini API呼び出しエラー: {e}")
        return {}

# ----------------------------------------------------------------
# Module: Main Application (UI)
# ----------------------------------------------------------------
def main():
    """
    防災スピーカー音圧可視化マップのメインアプリケーション関数。
    自動最適配置アルゴリズム、Gemini API 呼び出し、各種操作パネルを統合。
    """
    st.set_page_config(page_title="防災スピーカー音圧可視化マップ", layout="wide")
    st.title("防災スピーカー音圧可視化マップ")
    
    # セッションステートの初期化
    if "map_center" not in st.session_state:
        st.session_state.map_center = [34.25741795269067, 133.20450105700033]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 14
    if "speakers" not in st.session_state:
        st.session_state.speakers = [[34.25741795269067, 133.20450105700033, [0.0, 90.0]]]
    if "measurements" not in st.session_state:
        st.session_state.measurements = []
    if "heatmap_data" not in st.session_state:
        st.session_state.heatmap_data = None
    if "L0" not in st.session_state:
        st.session_state.L0 = 80
    if "r_max" not in st.session_state:
        st.session_state.r_max = 500
    
    # サイドバー：操作パネル
    with st.sidebar:
        st.header("操作パネル")
        
        # CSVアップロード
        uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        if uploaded_file:
            speakers, measurements = load_csv(uploaded_file)
            if speakers:
                st.session_state.speakers.extend(speakers)
            if measurements:
                st.session_state.measurements.extend(measurements)
            st.success("CSVファイルを読み込みました。")
            st.session_state.heatmap_data = None
        
        # スピーカー追加
        new_speaker = st.text_input("スピーカー追加 (緯度,経度,方向1,方向2...)", placeholder="例: 34.2579,133.2072,N,E")
        if st.button("スピーカー追加"):
            try:
                parts = new_speaker.split(",")
                lat, lon = float(parts[0]), float(parts[1])
                directions = [parse_direction(dir_str) for dir_str in parts[2:]]
                st.session_state.speakers.append([lat, lon, directions])
                st.session_state.heatmap_data = None
                st.success(f"スピーカーを追加しました: {lat}, {lon}, {directions}")
            except (ValueError, IndexError) as e:
                st.error("スピーカーの追加に失敗しました。形式が正しくない可能性があります。(緯度,経度,方向...)")
        
        # スピーカー削除機能
        if st.session_state.speakers:
            options = [f"{i}: ({spk[0]:.6f}, {spk[1]:.6f}) - 方向: {spk[2]}" 
                       for i, spk in enumerate(st.session_state.speakers)]
            selected_index = st.selectbox("削除するスピーカーを選択", list(range(len(options))),
                                          format_func=lambda i: options[i])
            if st.button("選択したスピーカーを削除"):
                try:
                    del st.session_state.speakers[selected_index]
                    st.session_state.heatmap_data = None
                    st.success("選択したスピーカーを削除しました")
                except Exception as e:
                    st.error(f"削除処理でエラーが発生しました: {e}")
        else:
            st.info("削除可能なスピーカーがありません。")
        
        # スピーカーリセット
        if st.button("スピーカーリセット"):
            st.session_state.speakers = []
            st.session_state.heatmap_data = None
            st.success("スピーカーをリセットしました")
        
        # パラメータ調整
        st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
        
        # 自動最適配置アルゴリズムのための目標音圧レベル（例：L0とL0-40の中間）
        target_default = st.session_state.L0 - 20
        target_level = st.slider("目標音圧レベル (dB)", st.session_state.L0 - 40, st.session_state.L0, target_default)
        if st.button("自動最適配置を実行"):
            # 表示範囲のグリッド（中心±0.01度）
            lat_min = st.session_state.map_center[0] - 0.01
            lat_max = st.session_state.map_center[0] + 0.01
            lon_min = st.session_state.map_center[1] - 0.01
            lon_max = st.session_state.map_center[1] + 0.01
            grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, 50), np.linspace(lon_min, lon_max, 50))
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
                st.success("自動最適配置アルゴリズムを実行しました")
            except Exception as e:
                st.error(f"最適配置アルゴリズムの実行中にエラーが発生しました: {e}")
        
        # Gemini API 呼び出し機能（プロンプトに地図情報を組み込む）
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_input("Gemini に問い合わせる内容")
        if st.button("Gemini API を実行"):
            full_prompt = generate_gemini_prompt(gemini_query)
            result = call_gemini_api(full_prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API の実行が完了しました")
    
    # メインパネル：地図とヒートマップの表示
    col1, col2 = st.columns([3, 1])
    with col1:
        # 表示範囲のグリッド生成（中心±0.01度）
        lat_min = st.session_state.map_center[0] - 0.01
        lat_max = st.session_state.map_center[0] + 0.01
        lon_min = st.session_state.map_center[1] - 0.01
        lon_max = st.session_state.map_center[1] + 0.01
        grid_lat, grid_lon = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))
    
        if st.session_state.heatmap_data is None:
            st.session_state.heatmap_data = calculate_heatmap(st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon)
        
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
        for spk in st.session_state.speakers:
            lat, lon, dirs = spk
            popup_text = f"<b>スピーカー</b>: ({lat:.6f}, {lon:.6f})<br><b>方向</b>: {dirs}"
            folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300)).add_to(m)
        
        if st.session_state.heatmap_data:
            HeatMap(
                st.session_state.heatmap_data,
                min_opacity=0.3,
                max_opacity=0.8,
                radius=15,
                blur=20
            ).add_to(m)
        
        st_folium(m, width=700, height=500)
    
    with col2:
        csv_data_speakers = export_csv(st.session_state.speakers, 
                                       ["スピーカー緯度", "スピーカー経度", "方向1", "方向2", "方向3"])
        st.download_button("スピーカーCSVダウンロード", csv_data_speakers, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        st.write("計測情報:", st.session_state.measurements)
        count = len(st.session_state.heatmap_data) if st.session_state.heatmap_data else 0
        st.write("ヒートマップデータの件数:", count)
    
    # 地図の下部にGemini APIの回答を表示する領域
    st.markdown("---")
    st.subheader("Gemini API の回答")
    if "gemini_result" in st.session_state:
        st.json(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答はまだありません。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
