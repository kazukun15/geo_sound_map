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

import matplotlib
import matplotlib.pyplot as plt

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
API_KEY = st.secrets["general"]["api_key"]  # secrets.toml の [general] セクションに設定
MODEL_NAME = "gemini-2.0-flash"

# ----------------------------------------------------------------
# CSV読み込み関連
# ----------------------------------------------------------------
def load_csv(file):
    """
    CSVファイルを読み込み、スピーカーの位置情報を抽出する関数。
    必須カラム: "latitude", "longitude"
    任意カラム: "label"（"施設名" や "名称" も利用可）
    戻り値は [[lat, lon, label], ...] の形式
    """
    try:
        df = pd.read_csv(file)
        speakers = []
        for idx, row in df.iterrows():
            try:
                lat_val = row.get("latitude")
                lon_val = row.get("longitude")
                if pd.isna(lat_val) or pd.isna(lon_val):
                    st.warning(f"行 {idx+1}: 'latitude' または 'longitude' が欠損しているためスキップ")
                    continue
                lat = float(lat_val)
                lon = float(lon_val)
                label = ""
                if "label" in df.columns and not pd.isna(row.get("label")):
                    label = str(row["label"]).strip()
                elif "施設名" in df.columns and not pd.isna(row.get("施設名")):
                    label = str(row["施設名"]).strip()
                elif "名称" in df.columns and not pd.isna(row.get("名称")):
                    label = str(row["名称"]).strip()
                speakers.append([lat, lon, label])
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
# 音圧計算
# ----------------------------------------------------------------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各スピーカーの音圧を計算し、全スピーカー分のパワーを合算して dB 値に変換する。
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, label = spk
        dlat = grid_lat - lat
        dlon = grid_lon - lon
        distance = np.sqrt((dlat * 111320)**2 + (dlon * 111320 * math.cos(math.radians(lat)))**2)
        distance[distance < 1] = 1
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
            val = sound_grid[i, j]
            if not np.isnan(val):
                heat_data.append([grid_lat[i, j], grid_lon[i, j], val])
    return heat_data

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """ヒートマップデータ計算をキャッシュして再計算を回避する。"""
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

# ----------------------------------------------------------------
# 等高線（コンターライン）表示用の関数
# ----------------------------------------------------------------
def add_contour_lines_to_map(m, grid_lat, grid_lon, speakers, L0, r_max, levels=None):
    """
    ヒートマップの代わりに、コンターライン（等高線）を Folium 上に表示する例。
    1. compute_sound_grid で音圧を計算
    2. matplotlib の contour を用いて線分を取得
    3. Folium PolyLine で地図上に描画
    """
    if levels is None:
        # 例として、4つのレベルに区切る
        # L0 が80dBの場合、[70, 60, 50, 40]あたり
        # 必要に応じて調整してください
        levels = [L0 - 10, L0 - 20, L0 - 30, L0 - 40]

    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    
    # matplotlib で contour を計算する
    fig, ax = plt.subplots()
    # grid_lon, grid_lat は 2D配列
    # contour の引数は (X, Y, Z) の順で X=lon, Y=lat とする場合は
    #    c = ax.contour(grid_lon, grid_lat, sound_grid, levels=levels)
    # しかし lat-lon の順序で引数にするなら (grid_lat, grid_lon, sound_grid)
    # ここでは x=lon, y=lat として contour する例
    c = ax.contour(grid_lon, grid_lat, sound_grid, levels=levels)
    
    # c.allsegs[i] はレベル i に対する線分のリスト
    # seg は Nx2 の配列で [x, y] = [lon, lat]
    colors = ["red", "blue", "green", "purple", "orange"]
    for level_idx, level_segs in enumerate(c.allsegs):
        color = colors[level_idx % len(colors)]
        for seg in level_segs:
            # seg: Nx2 -> [ [lon1, lat1], [lon2, lat2], ... ]
            # Folium では lat-lon の順序が必要
            coords = [[p[1], p[0]] for p in seg]
            folium.PolyLine(coords, color=color, weight=2).add_to(m)
    
    plt.close(fig)  # 不要な図を閉じる

# ----------------------------------------------------------------
# Gemini API 関連
# ----------------------------------------------------------------
def generate_gemini_prompt(user_query):
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
    Gemini のレスポンステキストからスピーカー提案を抽出する。
    例: "## スピーカー増設提案\n緯度 34.254000, 経度 133.208000, 方向 270, ラベル 役場"
    戻り値: [[lat, lon, label], ...]
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
        except Exception:
            continue
    return results

def add_speaker_proposals_from_gemini():
    if "gemini_result" not in st.session_state or not st.session_state.gemini_result:
        st.error("Gemini API の回答がありません。")
        return
    response_text = st.session_state.gemini_result
    proposals = extract_speaker_proposals(response_text)
    if proposals:
        added_count = 0
        for proposal in proposals:
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

# ----------------------------------------------------------------
# メインアプリ
# ----------------------------------------------------------------
def main():
    st.title("防災スピーカー音圧可視化マップ（ヒートマップ or 等高線 + Gemini API）")
    
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
    if "gemini_result" not in st.session_state:
        st.session_state.gemini_result = None
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = None
    
    with st.sidebar:
        st.header("操作パネル")
        
        # CSVファイルアップロード
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
        
        # スピーカー手動追加
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
            options = [f"{i}: ({s[0]:.6f}, {s[1]:.6f}) - {s[2]}" for i, s in enumerate(st.session_state.speakers)]
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
        
        # 編集フォーム
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
            st.success("スピーカーをリセットしました")
        
        # 音響パラメータ
        st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
        
        # ヒートマップ or 等高線表示を選択
        display_mode = st.radio("表示モードを選択", ["HeatMap", "Contour Lines"])
        
        # Gemini API
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_input("Gemini に問い合わせる内容")
        if st.button("Gemini API を実行"):
            full_prompt = generate_gemini_prompt(gemini_query)
            result = call_gemini_api(full_prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API 実行完了")
        
        # Gemini 提案スピーカー追加
        if st.session_state.get("gemini_result"):
            if st.button("Gemini 提案をスピーカーに追加"):
                add_speaker_proposals_from_gemini()
    
    # メインパネル：地図とヒートマップ or 等高線の表示
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
        
        # ヒートマップ or コンターライン計算
        if display_mode == "HeatMap":
            if st.session_state.heatmap_data is None:
                st.session_state.heatmap_data = cached_calculate_heatmap(
                    st.session_state.speakers,
                    st.session_state.L0,
                    st.session_state.r_max,
                    grid_lat,
                    grid_lon
                )
        else:
            # コンターラインの場合は特にキャッシュを使わず毎回計算
            # （キャッシュしてもOKです）
            st.session_state.heatmap_data = None
        
        # Folium地図作成
        m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom)
        
        # スピーカーをマーカーで表示
        for spk in st.session_state.speakers:
            lat, lon, label = spk
            popup_text = f"<b>スピーカー</b>: ({lat:.6f}, {lon:.6f})"
            if label:
                popup_text += f"<br><b>ラベル</b>: {label}"
            folium.Marker(location=[lat, lon], popup=folium.Popup(popup_text, max_width=300)).add_to(m)
        
        # ヒートマップ or コンターラインを描画
        if display_mode == "HeatMap":
            if st.session_state.heatmap_data:
                HeatMap(st.session_state.heatmap_data, min_opacity=0.3, max_opacity=0.8,
                        radius=15, blur=20).add_to(m)
        else:
            # コンターライン表示
            add_contour_lines_to_map(m, grid_lat, grid_lon, st.session_state.speakers,
                                     st.session_state.L0, st.session_state.r_max,
                                     levels=None)
        
        st_folium(m, width=700, height=500)
    
    with col2:
        # CSVダウンロード
        csv_data = export_csv(st.session_state.speakers)
        st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        if st.session_state.heatmap_data:
            count = len(st.session_state.heatmap_data)
        else:
            count = 0
        st.write("ヒートマップデータの件数:", count)
    
    st.markdown("---")
    st.subheader("Gemini API の回答（テキスト表示）")
    if st.session_state.gemini_result:
        st.text(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答はまだありません。")

# ----------------------------------------------------------------
# コンターライン描画関数
# ----------------------------------------------------------------
def add_contour_lines_to_map(m, grid_lat, grid_lon, speakers, L0, r_max, levels=None):
    """
    コンターライン（等高線）を Folium 上に表示する。
    1. compute_sound_grid で音圧を計算
    2. matplotlib の contour で線分を取得
    3. Folium PolyLine として追加
    """
    if levels is None:
        # 例: 4つのレベル
        levels = [L0 - 10, L0 - 20, L0 - 30, L0 - 40]
    
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    
    fig, ax = plt.subplots()
    # grid_lon, grid_lat, sound_grid の順で contour する
    c = ax.contour(grid_lon, grid_lat, sound_grid, levels=levels)
    
    colors = ["red", "blue", "green", "purple", "orange"]
    for level_idx, level_segs in enumerate(c.allsegs):
        color = colors[level_idx % len(colors)]
        for seg in level_segs:
            coords = [[p[1], p[0]] for p in seg]  # Foliumは [lat, lon] 順
            folium.PolyLine(coords, color=color, weight=2).add_to(m)
    
    plt.close(fig)  # 不要な図を閉じる

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
