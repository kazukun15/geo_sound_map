import os
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import math
import io
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
API_KEY = st.secrets["general"]["api_key"]  # secrets.toml の [general] セクションで設定
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
    CSVファイルを読み込み、スピーカーの位置情報を抽出する関数。
    必須カラム: "latitude", "longitude"
    任意カラム: "label"（"施設名" や "名称" も利用可能）
    
    戻り値は [[lat, lon, label], ...] の形式
    """
    try:
        df = pd.read_csv(file)
        speakers = []
        for idx, row in df.iterrows():
            try:
                if not pd.isna(row.get("latitude")) and not pd.isna(row.get("longitude")):
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                    label = ""
                    if "label" in df.columns and not pd.isna(row.get("label")):
                        label = str(row["label"]).strip()
                    elif "施設名" in df.columns and not pd.isna(row.get("施設名")):
                        label = str(row["施設名"]).strip()
                    elif "名称" in df.columns and not pd.isna(row.get("名称")):
                        label = str(row["名称"]).strip()
                    speakers.append([lat, lon, label])
                else:
                    st.warning(f"行 {idx+1}: 'latitude' または 'longitude' が欠損しているためスキップ")
            except Exception as e:
                st.warning(f"行 {idx+1} の読み込みに失敗しました: {e}")
        return speakers
    except Exception as e:
        st.error(f"CSV読み込み全体でエラー: {e}")
        return []

def export_csv(data):
    """
    スピーカー情報（[[lat, lon, label], ...]）を CSV 形式に変換する関数。
    CSV の列は ["latitude", "longitude", "label"] とする。
    """
    rows = []
    for entry in data:
        if len(entry) == 3:
            lat, lon, label = entry
            rows.append({
                "latitude": lat,
                "longitude": lon,
                "label": label
            })
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "label"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ----------------------------------------------------------------
# Module: Heatmap Calculation & Sound Grid Utilities (パフォーマンス最適化)
# ----------------------------------------------------------------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各スピーカーの音圧を計算し、全スピーカー分のパワーを合算して dB 値に変換する。
    speakers: [[lat, lon, label], ...]
    L0: 初期音圧レベル (dB)
    r_max: 最大伝播距離 (m)
    grid_lat, grid_lon: 各グリッド点の緯度・経度（2D配列）
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, label = spk
        dlat = grid_lat - lat
        dlon = grid_lon - lon
        distance = np.sqrt((dlat * 111320)**2 + (dlon * 111320 * math.cos(math.radians(lat)))**2)
        distance[distance < 1] = 1
        # 単純な減衰モデル: L0 - 20 log10(distance)
        p_db = L0 - 20 * np.log10(distance)
        p_db[distance > r_max] = -999
        valid = p_db > -999
        power_sum[valid] += 10 ** (p_db[valid] / 10)
    sound_grid = np.full_like(power_sum, np.nan, dtype=float)
    valid = power_sum > 0
    sound_grid[valid] = 10 * np.log10(power_sum[valid])
    return sound_grid

def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """
    ヒートマップ表示用のデータリストを作成する関数。
    """
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    heat_data = []
    for i in range(Nx):
        for j in range(Ny):
            if not np.isnan(sound_grid[i, j]):
                heat_data.append([grid_lat[i, j], grid_lon[i, j], sound_grid[i, j]])
    return heat_data

import streamlit as st  # 既にインポート済みですが、キャッシュ用に再記載
@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """ヒートマップデータ計算をキャッシュして再計算を回避する。"""
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

# ----------------------------------------------------------------
# Gemini API 関連
# ----------------------------------------------------------------
def generate_gemini_prompt(user_query):
    """
    ユーザーの問い合わせと現在のスピーカー配置、音圧分布情報を元に改善案を提案するプロンプトを生成します。
    以下の条件を加味してください：
      - 海など設置困難な場所は除外
      - スピーカー同士は300m以上離れていること
      - 座標表記は「緯度 xxx.xxxxxx, 経度 yyy.yyyyyy」で統一
    """
    speakers = st.session_state.speakers if "speakers" in st.session_state else []
    if speakers:
        speaker_info = "配置されているスピーカー:\n" + "\n".join(
            f"{i+1}. 緯度: {s[0]:.6f}, 経度: {s[1]:.6f}, ラベル: {s[2]}"
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
# Gemini API の提案からスピーカーを追加する機能
# ----------------------------------------------------------------
def extract_speaker_proposals(response_text):
    """
    Gemini のレスポンステキストから、スピーカー追加の提案を抽出する関数。
    例として、レスポンス内に「緯度 34.254000, 経度 133.208000, 方向 270, ラベル 役場」
    のような記述があることを想定しています。
    戻り値は [[lat, lon, label], ...] の形式（方向情報は今回は省略）
    """
    pattern = r"緯度\s*([-\d]+\.\d+)\s*,\s*経度\s*([-\d]+\.\d+)(?:\s*,\s*(?:方向\s*[^\d]*([\d\.]+))?)?(?:\s*,\s*(?:ラベル\s*([^\n]+)))?"
    proposals = re.findall(pattern, response_text)
    results = []
    for lat_str, lon_str, direction, label in proposals:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            label = label.strip() if label else ""
            results.append([lat, lon, label])
        except Exception as e:
            continue
    return results

def add_speaker_proposals_from_gemini():
    """
    Gemini API の提案からスピーカー情報を抽出し、既存のスピーカーリストに追加する。
    """
    if "gemini_result" not in st.session_state or not st.session_state.gemini_result:
        st.error("Gemini API の回答がありません。")
        return
    response_text = st.session_state.gemini_result
    proposals = extract_speaker_proposals(response_text)
    if proposals:
        added_count = 0
        for proposal in proposals:
            # すでに同一の座標がある場合は追加しない
            if not any(abs(proposal[0] - s[0]) < 1e-6 and abs(proposal[1] - s[1]) < 1e-6 for s in st.session_state.speakers):
                st.session_state.speakers.append(proposal)
                added_count += 1
        if added_count > 0:
            st.success(f"Geminiの提案から {added_count} 件のスピーカー情報を追加しました。")
            st.session_state.heatmap_data = None
        else:
            st.info("Geminiの提案から新たなスピーカー情報は見つかりませんでした。")
    else:
        st.info("Geminiの提案からスピーカー情報の抽出に失敗しました。")

# ----------------------------------------------------------------
# ThreadPoolExecutor（非同期処理）
# ----------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=2)

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """ヒートマップデータ計算をキャッシュして、再計算を回避する。"""
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

# ----------------------------------------------------------------
# Main Application (UI) – メインパネル・UI改善
# ----------------------------------------------------------------
def main():
    st.title("防災スピーカー音圧可視化マップ（複数スピーカー＋Gemini API）")
    
    # セッション初期化
    if "map_center" not in st.session_state:
        st.session_state.map_center = [34.25741795269067, 133.20450105700033]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 14
    if "speakers" not in st.session_state:
        st.session_state.speakers = [
            [34.25741795269067, 133.20450105700033, "初期スピーカーA"],
            [34.2574617056359, 133.204487449849, "初期スピーカーB"]
        ]
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
                new_speakers = load_csv(uploaded_file)
                if new_speakers:
                    st.session_state.speakers.extend(new_speakers)
                    st.session_state.heatmap_data = None
                    st.success(f"CSVから {len(new_speakers)} 件のスピーカー情報を追加しました。")
                else:
                    st.error("CSVに正しいデータが見つかりませんでした。")
        
        # スピーカー手動追加（(latitude,longitude,label) 形式）
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
                    st.success(f"スピーカー追加成功: 緯度 {lat}, 経度 {lon}, ラベル: {label}")
                except Exception as e:
                    st.error(f"スピーカー追加エラー: {e}")
        
        # スピーカー削除・編集
        if st.session_state.speakers:
            options = [f"{i}: ({s[0]:.6f}, {s[1]:.6f}) - ラベル: {s[2]}" for i, s in enumerate(st.session_state.speakers)]
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
                new_label = st.text_input("新しいラベル", value=spk[2])
                submitted = st.form_submit_button("編集保存")
                if submitted:
                    try:
                        lat_val = float(new_lat)
                        lon_val = float(new_lon)
                        st.session_state.speakers[st.session_state.edit_index] = [lat_val, lon_val, new_label]
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
        
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_input("Gemini に問い合わせる内容")
        if st.button("Gemini API を実行"):
            full_prompt = generate_gemini_prompt(gemini_query)
            result = call_gemini_api(full_prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API 実行完了")
        
        # Gemini の提案からスピーカー追加ボタン
        if st.session_state.get("gemini_result"):
            if st.button("Gemini 提案をスピーカーに追加"):
                add_speaker_proposals_from_gemini()
    
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
            lat, lon, label = spk
            popup_text = f"<b>スピーカー</b>: ({lat:.6f}, {lon:.6f})"
            if label:
                popup_text += f"<br><b>ラベル</b>: {label}"
            folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300)).add_to(m)
        if st.session_state.heatmap_data:
            HeatMap(st.session_state.heatmap_data, min_opacity=0.3, max_opacity=0.8, radius=15, blur=20).add_to(m)
        st_folium(m, width=700, height=500)
    with col2:
        csv_data = export_csv(st.session_state.speakers)
        st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        count = len(st.session_state.heatmap_data) if st.session_state.heatmap_data else 0
        st.write("ヒートマップデータの件数:", count)
    
    st.markdown("---")
    st.subheader("Gemini API の回答（テキスト表示）")
    if st.session_state.gemini_result:
        st.text(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答はまだありません。")

# ----------------------------------------------------------------
# Gemini API の提案からスピーカーを追加する関数
# ----------------------------------------------------------------
def extract_speaker_proposals(response_text):
    """
    Gemini のレスポンステキストからスピーカー提案を抽出する。
    例: "## スピーカー増設提案\n緯度 34.254000, 経度 133.208000, 方向 270, ラベル 役場"
    ※ ここでは「緯度」と「経度」と、任意で「ラベル」を抽出します。
    戻り値は [[lat, lon, label], ...] の形式。
    """
    pattern = r"緯度\s*([-\d]+\.\d+)\s*,\s*経度\s*([-\d]+\.\d+)(?:\s*,\s*(?:方向\s*[^\d]*([\d\.]+))?)?(?:\s*,\s*(?:ラベル\s*([^\n]+))?)?"
    proposals = re.findall(pattern, response_text)
    results = []
    for lat_str, lon_str, direction, label in proposals:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            label = label.strip() if label else ""
            results.append([lat, lon, label])
        except Exception as e:
            continue
    return results

def add_speaker_proposals_from_gemini():
    """
    Gemini の提案からスピーカー情報を抽出し、既存のスピーカーリストに追加する。
    """
    if "gemini_result" not in st.session_state or not st.session_state.gemini_result:
        st.error("Gemini API の回答がありません。")
        return
    response_text = st.session_state.gemini_result
    proposals = extract_speaker_proposals(response_text)
    if proposals:
        added_count = 0
        for proposal in proposals:
            # すでに同一の座標がある場合は追加しない
            if not any(abs(proposal[0] - s[0]) < 1e-6 and abs(proposal[1] - s[1]) < 1e-6 for s in st.session_state.speakers):
                st.session_state.speakers.append(proposal)
                added_count += 1
        if added_count > 0:
            st.success(f"Gemini の提案から {added_count} 件のスピーカー情報を追加しました。")
            st.session_state.heatmap_data = None
        else:
            st.info("Gemini の提案から新たなスピーカー情報は見つかりませんでした。")
    else:
        st.info("Gemini の提案からスピーカー情報の抽出に失敗しました。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
