import os
import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
import math
import io
import requests
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt  # ※等高線用の処理などで使用する場合もあるが、ここではHeatmapを利用

# st.set_page_config() は必ず最初に呼び出す
st.set_page_config(page_title="防災スピーカー音圧可視化マップ", layout="wide")

# ---------- Custom CSS for UI styling ----------
custom_css = """
<style>
body { font-family: 'Helvetica', sans-serif; }
h1, h2, h3, h4, h5, h6 { color: #333333; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-weight: bold; color: #4CAF50;
}
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
    CSVファイルを読み込み、スピーカーの位置情報を抽出する。
    必須カラム: "latitude", "longitude"
    任意カラム: "label"（"施設名" や "名称" も利用可能）
    戻り値は [[lat, lon, label], ...] の形式。
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
    ※ 新規スピーカーの "new" フラグは出力しません。
    """
    rows = []
    for entry in data:
        if len(entry) >= 3:
            lat, lon, label = entry[0], entry[1], entry[2]
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
# 音圧計算関連
# ----------------------------------------------------------------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各スピーカーの音圧を計算し、全スピーカー分のパワーを合算して dB 値に変換する。
    speakers: [[lat, lon, label], ...]
    L0: 初期音圧レベル (dB)
    r_max: 最大伝播距離 (m)
    grid_lat, grid_lon: 2D配列の各グリッド点の緯度・経度
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
    戻り値は DataFrame（columns: latitude, longitude, value）
    """
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                data.append({
                    "latitude": grid_lat[i, j],
                    "longitude": grid_lon[i, j],
                    "value": val
                })
    return pd.DataFrame(data)

# キャッシュ付きヒートマップ
@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

# ----------------------------------------------------------------
# Pydeck 用にスピーカー DataFrame を作成する関数
# ----------------------------------------------------------------
def get_speaker_df(speakers):
    """
    内部リスト (speakers) を DataFrame に変換する。
    新規スピーカー（4要素目が "new"）は new フラグ True、それ以外は False とする。
    """
    data = []
    for spk in speakers:
        if len(spk) >= 4 and spk[3] == "new":
            new_flag = True
        else:
            new_flag = False
        data.append({
            "latitude": spk[0],
            "longitude": spk[1],
            "label": spk[2],
            "new": new_flag
        })
    return pd.DataFrame(data)

# ----------------------------------------------------------------
# Gemini API 関連
# ----------------------------------------------------------------
def generate_gemini_prompt(user_query):
    """
    Gemini へのプロンプトでは、必ず「緯度: 34.255500, 経度: 133.207000」の形式で出力するように指示してください。
    ユーザーの問い合わせと現在のスピーカー配置、音圧分布情報を元に改善案を提案するプロンプトを生成します。
    条件：
      - 海など設置困難な場所は除外
      - スピーカー同士は300m以上離れていること
      - 座標表記は「緯度: 34.255500, 経度: 133.207000」で統一
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
        "【座標表記形式】 緯度: 34.255500, 経度: 133.207000 で統一してください。"
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
    Gemini のレスポンステキストからスピーカー提案を抽出する関数です。
    例:
      **新規スピーカーC:**
          *   緯度: 34.259000, 経度: 133.201500
          *   理由: 既存スピーカーA/Bから北西方向に約400m離れた、比較的平坦な土地。
      **新規スピーカーD:**
          *   緯度: 34.255500, 経度: 133.207000
          *   理由: 既存スピーカーA/Bから南東方向に約500m離れた、開けた場所。
    この関数は、緯度・経度（および任意のラベル）が含まれていれば抽出し、
    [[lat, lon, label, "new"], ...] の形式で返します。
    """
    pattern = r"(?:緯度[:：]?\s*)([-\d]+\.\d+)[,、\s]+(?:経度[:：]?\s*)([-\d]+\.\d+)(?:[,、\s]+(?:方向[:：]?\s*[-\d\.]+))?(?:[,、\s]+(?:ラベル[:：]?\s*([^\n]+))?)?"
    proposals = re.findall(pattern, response_text)
    results = []
    for lat_str, lon_str, label in proposals:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            label = label.strip() if label else ""
            results.append([lat, lon, label, "new"])
        except Exception:
            continue
    return results

def add_speaker_proposals_from_gemini():
    """
    Gemini の回答内に新設の緯度・経度が含まれている場合、
    それらを抽出して既存のスピーカーリストに追加する。
    新規に追加されたスピーカーは "new" フラグ付きで保存する。
    """
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
            st.success(f"Gemini の回答から {added_count} 件の新規スピーカー情報を追加しました。")
            st.session_state.heatmap_data = None
        else:
            st.info("Gemini の回答から新たなスピーカー情報は見つかりませんでした。")
    else:
        st.info("Gemini の回答からスピーカー情報の抽出に失敗しました。")

# ----------------------------------------------------------------
# ThreadPoolExecutor（非同期処理）
# ----------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=2)

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

# ----------------------------------------------------------------
# Pydeck 用にスピーカー DataFrame を作成
# ----------------------------------------------------------------
def get_speaker_df(speakers):
    data = []
    for spk in speakers:
        if len(spk) >= 4 and spk[3] == "new":
            new_flag = True
        else:
            new_flag = False
        data.append({
            "latitude": spk[0],
            "longitude": spk[1],
            "label": spk[2],
            "new": new_flag
        })
    return pd.DataFrame(data)

# ----------------------------------------------------------------
# Main Application (UI) – Pydeck 表示版
# ----------------------------------------------------------------
def main():
    st.title("防災スピーカー音圧可視化マップ (Pydeck版)")
    
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

    # サイドバー：操作パネル
    with st.sidebar:
        st.header("操作パネル")
        
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
            st.success("スピーカーをリセットしました")
        
        st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
        
        display_mode = st.radio("表示モードを選択", ["HeatMap", "Markers Only"])
        
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_input("Gemini に問い合わせる内容")
        if st.button("Gemini API を実行"):
            full_prompt = generate_gemini_prompt(gemini_query)
            result = call_gemini_api(full_prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API 実行完了")
            # 自動的にGemini回答から新規スピーカー追加
            add_speaker_proposals_from_gemini()
    
    # Pydeck 表示用データ作成
    speaker_df = get_speaker_df(st.session_state.speakers)
    # マーカーの色は、new=True の場合は赤、Falseの場合は青
    if not speaker_df.empty:
        speaker_df["color"] = speaker_df["new"].apply(lambda x: [255, 0, 0, 200] if x else [0, 0, 255, 200])
    
    # ヒートマップデータの作成（HeatMap表示モードの場合）
    # グリッドを st.session_state.map_center ±0.01 度で作成
    lat_min = st.session_state.map_center[0] - 0.01
    lat_max = st.session_state.map_center[0] + 0.01
    lon_min = st.session_state.map_center[1] - 0.01
    lon_max = st.session_state.map_center[1] + 0.01
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, 100),
        np.linspace(lon_min, lon_max, 100)
    )
    heat_df = None
    if display_mode == "HeatMap":
        if st.session_state.heatmap_data is None:
            st.session_state.heatmap_data = cached_calculate_heatmap(
                st.session_state.speakers,
                st.session_state.L0,
                st.session_state.r_max,
                grid_lat,
                grid_lon
            )
        heat_df = st.session_state.heatmap_data

    # Pydeck レイヤー作成
    layers = []
    # スピーカー表示用：ScatterplotLayer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=speaker_df,
        get_position=["longitude", "latitude"],
        get_radius=200,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )
    layers.append(scatter_layer)
    
    # ヒートマップ表示用：HeatmapLayer（選択時のみ）
    if display_mode == "HeatMap" and heat_df is not None and not heat_df.empty:
        heat_layer = pdk.Layer(
            "HeatmapLayer",
            data=heat_df,
            get_position=["longitude", "latitude"],
            get_weight="value",
            radiusPixels=50,
            aggregation=pdk.types.String("SUM"),
        )
        layers.append(heat_layer)
    
    # 初期のビュー状態
    view_state = pdk.ViewState(
        latitude=st.session_state.map_center[0],
        longitude=st.session_state.map_center[1],
        zoom=st.session_state.map_zoom,
        pitch=0,
    )
    
    deck = pdk.Deck(
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{label}"},
    )
    st.pydeck_chart(deck)
    
    # CSVダウンロード用
    csv_data = export_csv(st.session_state.speakers)
    st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        if heat_df is not None:
            st.write("ヒートマップデータ件数:", len(heat_df))
        else:
            st.write("ヒートマップデータなし")
    
    st.markdown("---")
    st.subheader("Gemini API の回答（テキスト表示）")
    if st.session_state.gemini_result:
        st.text(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答はまだありません。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
