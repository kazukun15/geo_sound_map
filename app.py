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

# st.set_page_config() は必ず最初に呼び出す
st.set_page_config(page_title="防災スピーカー音圧可視化マップ", layout="wide")

# ---------- Custom CSS for UI styling ----------
custom_css = """
<style>
/* 全体のフォント */
body {
    font-family: 'Helvetica', sans-serif;
}

/* ヘッダーの色 */
h1, h2, h3, h4, h5, h6 {
    color: #333333;
}

/* ボタンのスタイル */
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    border-radius: 8px;
    cursor: pointer;
}
div.stButton > button:hover {
    background-color: #45a049;
}

/* テキスト入力とテキストエリアのスタイル */
div.stTextInput>div>input, div.stTextArea>div>textarea {
    font-size: 16px;
    padding: 8px;
}

/* サイドバーのタイトル */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-weight: bold;
    color: #4CAF50;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# ---------- End Custom CSS ----------

# ------------------------------------------------------------------
# 定数／設定（APIキー、モデル）
# ------------------------------------------------------------------
API_KEY = st.secrets["general"]["api_key"]  # secrets.toml に [general] セクションで設定してください
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
    入力文字列から方向（度数）に変換する関数。
    "N", "E", "S", "W"（大文字・小文字問わず）も数字も受け付けます。
    例: "N" -> 0, "sw" -> 225, "45" -> 45.0
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
    
    Returns:
      speakers: [[lat, lon, [dir1, dir2, ...]], ...]
      measurements: [[lat, lon, db], ...]
    """
    try:
        df = pd.read_csv(file)
        speakers, measurements = [], []
        for _, row in df.iterrows():
            if not pd.isna(row.get("スピーカー緯度")):
                lat, lon = row["スピーカー緯度"], row["スピーカー経度"]
                directions = [parse_direction(row.get(f"方向{i}", "")) 
                              for i in range(1, 4) if not pd.isna(row.get(f"方向{i}"))]
                speakers.append([lat, lon, directions])
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
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各グリッド点における音圧レベルを計算し、2D配列（sound_grid）として返す関数。
    各スピーカーの影響は、距離と方向性（コサイン補正＋最低 0.3 倍）に基づいて計算されます。
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    
    # grid_lat, grid_lon は2D配列（度単位）
    for spk in speakers:
        lat, lon, dirs = spk
        dlat = grid_lat - lat
        dlon = grid_lon - lon
        # 経度補正（緯度依存）
        distance = np.sqrt((dlat * 111320)**2 + (dlon * 111320 * np.cos(np.radians(lat)))**2)
        distance[distance < 1] = 1
        # 各グリッド点への方位（度単位）
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
    """
    ヒートマップ用のデータリストを作成する関数。
    """
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
    """
    目標音圧レベルとの差の二乗平均誤差（MSE）を計算する関数。
    """
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    valid = ~np.isnan(sound_grid)
    mse = np.mean((sound_grid[valid] - target)**2)
    return mse

def optimize_speaker_placement(speakers, target, L0, r_max, grid_lat, grid_lon, iterations=10, delta=0.0001):
    """
    各スピーカーの位置を微調整し、目標音圧レベルとの差（二乗平均誤差）を最小化する自動最適配置アルゴリズム。
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
    以下の条件と地形情報を考慮し、最適なスピーカー配置案を提案してください。

    【条件】
    - スピーカーは、被災地域全体に均一に音声を届ける必要がある。
    - スピーカー同士は、お互いの干渉を避けるために、原則として300m以上離れるように配置する。
    - 各スピーカーは、設置場所の地形（山、谷、海岸、島、樹林など）や障害物を考慮し、最適な方向に向ける必要がある。
    - 島や複雑な地形がある場合は、斜面や海岸線などの特性を反映して、スピーカー配置が実際の音の伝播に与える影響を最小限にする配置を考える。

    【出力形式】
    - 提案する各スピーカーの配置は、必ず以下の形式で出力してください。
      「緯度 xxx.xxxxxx, 経度 yyy.yyyyyy, 方向 Z」
      （例：緯度 34.254000, 経度 133.208000, 方向 270）
    - 各配置について、その設置理由や考慮した地形の特徴も併せて簡潔に説明してください。

    【ユーザーの問い合わせ】
    {user_query}

    上記の条件と情報に基づき、最も効果的なスピーカー配置案とその理由を、具体的かつ詳細に提案してください。
    """
    return generate_gemini_prompt.__doc__.format(user_query=user_query)

def call_gemini_api(query):
    """
    Gemini API に対してクエリを送信する関数。
    """
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": query}]
        }]
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Gemini API呼び出しエラー: {e}")
        return {}

# ----------------------------------------------------------------
# Module: Utility for extracting coordinates and direction from text
# ----------------------------------------------------------------
def extract_coords_and_dir_from_text(text):
    """
    説明文から「緯度 xxx.xxxxxx, 経度 yyy.yyyyyy, 方向 Z」形式の情報を抽出する関数。
    Zは数値として認識されます（例: 270）。
    コロン有り／無しの両方に対応します。
    例: "緯度 34.254000, 経度 133.208000, 方向 270"
         "34.284500, 133.104500, 方向 0"
         "緯度: 34.273000, 経度: 133.215000, 方向: 225"
    見つかった情報をリストで返す。例: [(34.254000, 133.208000, 270)]
    """
    pattern = r"(?:緯度[:：]?\s*)?([-\d]+\.\d+),\s*(?:経度[:：]?\s*)?([-\d]+\.\d+),\s*方向[:：]?\s*([-\d]+(?:\.\d+)?)"
    matches = re.findall(pattern, text)
    results = []
    for lat_str, lon_str, dir_str in matches:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            direction = parse_direction(dir_str)
            results.append((lat, lon, direction))
        except ValueError:
            continue
    return results

# ----------------------------------------------------------------
# Module: Utility for parsing speaker addition input
# ----------------------------------------------------------------
def parse_speaker_input(text):
    """
    入力欄に貼り付けられた固定形式の文字列から、(lat, lon, direction) を抽出する関数。
    対応例：
      "緯度 34.254000, 経度 133.208000, 方向 270"
      "緯度: 34.273000, 経度: 133.215000, 方向: 225"
      "34.284500, 133.104500, 方向 0"
    """
    pattern = r"^(?:緯度[:：]?\s*)?([-\d]+\.\d+),\s*(?:経度[:：]?\s*)?([-\d]+\.\d+),\s*方向[:：]?\s*([-\d]+(?:\.\d+)?)$"
    match = re.search(pattern, text)
    if match:
        lat_str, lon_str, dir_str = match.groups()
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            direction = parse_direction(dir_str)
            return lat, lon, direction
        except ValueError:
            return None
    return None

# ----------------------------------------------------------------
# Module: Main Application (UI)
# ----------------------------------------------------------------
def main():
    st.title("防災スピーカー音圧可視化マップ")
    
    # セッションステート初期化
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
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = None
    
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
        
        # スピーカー追加：固定形式の入力を想定
        new_speaker = st.text_input("スピーカー追加", 
                                    placeholder="例: 緯度 34.254000, 経度 133.208000, 方向 270\nまたは 34.284500, 133.104500, 方向 0\nまたは 緯度: 34.273000, 経度: 133.215000, 方向: 225")
        if st.button("スピーカー追加"):
            parsed = parse_speaker_input(new_speaker)
            if parsed:
                lat, lon, direction = parsed
                st.session_state.speakers.append([lat, lon, [direction]])
                st.session_state.heatmap_data = None
                st.success(f"スピーカーを追加しました: 緯度 {lat}, 経度 {lon}, 方向 {direction}")
            else:
                st.error("入力形式が正しくありません。形式は『緯度 34.254000, 経度 133.208000, 方向 270』などで入力してください。")
        
        # スピーカー削除・編集機能
        if st.session_state.speakers:
            options = [f"{i}: ({spk[0]:.6f}, {spk[1]:.6f}) - 方向: {spk[2]}" for i, spk in enumerate(st.session_state.speakers)]
            selected_index = st.selectbox("スピーカーを選択", list(range(len(options))),
                                          format_func=lambda i: options[i])
            col_del, col_edit = st.columns(2)
            with col_del:
                if st.button("選択したスピーカーを削除"):
                    try:
                        del st.session_state.speakers[selected_index]
                        st.session_state.heatmap_data = None
                        st.success("選択したスピーカーを削除しました")
                    except Exception as e:
                        st.error(f"削除処理でエラーが発生しました: {e}")
            with col_edit:
                if st.button("選択したスピーカーを編集"):
                    st.session_state.edit_index = selected_index
        else:
            st.info("スピーカーがありません。")
        
        # 編集フォーム
        if st.session_state.edit_index is not None:
            with st.form("edit_form"):
                spk = st.session_state.speakers[st.session_state.edit_index]
                new_lat = st.text_input("新しい緯度", value=str(spk[0]), key="edit_lat")
                new_lon = st.text_input("新しい経度", value=str(spk[1]), key="edit_lon")
                new_dirs = st.text_input("新しい方向（カンマ区切り、例: N,E,S,Wまたは数値）", 
                                         value=",".join(str(d) for d in spk[2]), key="edit_dirs")
                submitted = st.form_submit_button("編集内容を保存")
                if submitted:
                    try:
                        lat_val = float(new_lat)
                        lon_val = float(new_lon)
                        directions_val = [parse_direction(x) for x in new_dirs.split(",")]
                        st.session_state.speakers[st.session_state.edit_index] = [lat_val, lon_val, directions_val]
                        st.session_state.heatmap_data = None
                        st.success("スピーカー情報が更新されました")
                        st.session_state.edit_index = None
                    except Exception as e:
                        st.error(f"編集内容の保存に失敗しました: {e}")
        
        # スピーカーリセット（Gemini のレスポンスもクリア）
        if st.button("スピーカーリセット"):
            st.session_state.speakers = []
            st.session_state.heatmap_data = None
            st.session_state.gemini_result = None
            st.success("スピーカーをリセットしました")
        
        # パラメータ調整
        st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
        
        # 自動最適配置
        target_default = st.session_state.L0 - 20
        target_level = st.slider("目標音圧レベル (dB)", st.session_state.L0 - 40, st.session_state.L0, target_default)
        if st.button("自動最適配置を実行"):
            if st.session_state.speakers:
                lats = [s[0] for s in st.session_state.speakers]
                lons = [s[1] for s in st.session_state.speakers]
                margin = 0.005
                lat_min = min(lats) - margin
                lat_max = max(lats) + margin
                lon_min = min(lons) - margin
                lon_max = max(lons) + margin
            else:
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
                st.success("自動最適配置アルゴリズムを実行しました")
            except Exception as e:
                st.error(f"最適配置アルゴリズムの実行中にエラーが発生しました: {e}")
        
        # Gemini API 呼び出し
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_area("Gemini に問い合わせる内容", height=150,
                                    placeholder="ここに問い合わせ内容を入力してください")
        if st.button("Gemini API を実行"):
            # 以下のプロンプトは、島や地形情報を踏まえた最適なスピーカー配置案を提案するためのものです。
            full_prompt = (
                "あなたは、災害対策のために公共空間にスピーカーを配置する専門家です。以下の条件と地形情報を考慮し、最適なスピーカー配置案を提案してください。\n\n"
                "【条件】\n"
                "- スピーカーは、被災地域全体に均一に音声を届ける必要がある。\n"
                "- スピーカー同士は、お互いの干渉を避けるために、原則として300m以上離れるように配置する。\n"
                "- 各スピーカーは、設置場所の地形（山、谷、海岸、島、樹林など）や障害物を考慮し、最適な方向に向ける必要がある。\n"
                "- 島や複雑な地形がある場合は、斜面や海岸線などの特性を反映して、スピーカー配置が実際の音の伝播に与える影響を最小限にする配置を考える。\n\n"
                "【出力形式】\n"
                "- 提案する各スピーカーの配置は、必ず以下の形式で出力してください。\n"
                "  「緯度 xxx.xxxxxx, 経度 yyy.yyyyyy, 方向 Z」\n"
                "  （例：緯度 34.254000, 経度 133.208000, 方向 270）\n"
                "- 各配置について、その設置理由や考慮した地形の特徴も併せて簡潔に説明してください。\n\n"
                "【ユーザーの問い合わせ】\n"
                f"{gemini_query}\n\n"
                "上記の条件と情報に基づき、最も効果的なスピーカー配置案とその理由を、具体的かつ詳細に提案してください。"
            )
            result = call_gemini_api(full_prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API の実行が完了しました")
    
    # メインパネル：地図とヒートマップの表示（スピーカー全体の領域に合わせる）
    if st.session_state.speakers:
        lats = [s[0] for s in st.session_state.speakers]
        lons = [s[1] for s in st.session_state.speakers]
        margin = 0.005
        lat_min = min(lats) - margin
        lat_max = max(lats) + margin
        lon_min = min(lons) - margin
        lon_max = max(lons) + margin
        map_center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
    else:
        lat_min = st.session_state.map_center[0] - 0.01
        lat_max = st.session_state.map_center[0] + 0.01
        lon_min = st.session_state.map_center[1] - 0.01
        lon_max = st.session_state.map_center[1] + 0.01
        map_center = st.session_state.map_center
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
    
    m = folium.Map(location=map_center, zoom_start=st.session_state.map_zoom)
    for spk in st.session_state.speakers:
        lat, lon, dirs = spk
        popup_text = f"<b>スピーカー</b>: ({lat:.6f}, {lon:.6f})<br><b>方向</b>: {dirs}"
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)
    
    if st.session_state.heatmap_data:
        HeatMap(
            st.session_state.heatmap_data,
            min_opacity=0.3,
            max_opacity=0.8,
            radius=15,
            blur=20
        ).add_to(m)
    st_folium(m, width=700, height=500)
    
    with st.columns([3, 1])[1]:
        csv_data_speakers = export_csv(
            st.session_state.speakers,
            ["スピーカー緯度", "スピーカー経度", "方向1", "方向2", "方向3"]
        )
        st.download_button("スピーカーCSVダウンロード", csv_data_speakers, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        st.write("計測情報:", st.session_state.measurements)
        count = len(st.session_state.heatmap_data) if st.session_state.heatmap_data else 0
        st.write("ヒートマップデータの件数:", count)
    
    # -------------------------------------------------------
    # Gemini API の回答表示 (説明部分 & JSON 全体)
    # -------------------------------------------------------
    st.markdown("---")
    st.subheader("Gemini API の回答（説明部分 & JSON）")
    if "gemini_result" in st.session_state:
        result = st.session_state.gemini_result
        
        # candidates[0].content.parts[0].text を抽出して説明部分として表示
        explanation_text = ""
        try:
            explanation_text = result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            pass
        
        if explanation_text:
            st.markdown("#### 説明部分")
            st.write(explanation_text)
            
            # 固定形式の情報抽出：例「緯度 34.254000, 経度 133.208000, 方向 270」
            coords_dirs = extract_coords_and_dir_from_text(explanation_text)
            if coords_dirs:
                st.markdown("##### 以下の座標と方向を検出しました。地図に追加します。")
                for (lat, lon, direction) in coords_dirs:
                    if not any(abs(lat - s[0]) < 1e-6 and abs(lon - s[1]) < 1e-6 for s in st.session_state.speakers):
                        st.session_state.speakers.append([lat, lon, [direction]])
                        st.write(f"- 緯度: {lat}, 経度: {lon}, 方向: {direction} を追加")
                st.session_state.heatmap_data = None
            else:
                st.info("説明文から固定形式の座標と方向は検出されませんでした。")
        else:
            st.error("説明部分の抽出に失敗しました。JSON構造を確認してください。")
        
        st.markdown("#### JSON 全体")
        st.json(result)
    else:
        st.info("Gemini API の回答はまだありません。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
