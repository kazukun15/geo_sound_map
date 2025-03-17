import os
import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
import math
import io
import requests
import re
import random
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# ----------------------------------------
# 初期設定とカスタムスタイル
# ----------------------------------------
st.set_page_config(page_title="防災スピーカー音圧可視化マップ (Pydeck)", layout="wide")

CUSTOM_CSS = """
<style>
body { font-family: 'Helvetica', sans-serif; }
h1, h2, h3, h4, h5, h6 { color: #333333; }
div.stButton > button {
    background-color: #4CAF50; color: white; border: none;
    padding: 10px 24px; font-size: 16px; border-radius: 8px; cursor: pointer;
}
div.stButton > button:hover { background-color: #45a049; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-weight: bold; color: #4CAF50;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------
# 定数／API設定
# ----------------------------------------
API_KEY = st.secrets["general"]["api_key"]  # secrets.toml の [general] に設定
MODEL_NAME = "gemini-2.0-flash"

# ----------------------------------------
# CSV読み込み／書き出し関連
# ----------------------------------------
def load_csv(file) -> list:
    """CSVからスピーカー情報を読み込み、[[lat, lon, label], ...] を返す"""
    try:
        df = pd.read_csv(file)
        speakers = []
        for idx, row in df.iterrows():
            try:
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
            except Exception as e:
                st.warning(f"行 {idx+1} 読み込み失敗: {e}")
        return speakers
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return []

def export_csv(data) -> bytes:
    """スピーカー情報をCSVに変換（出力カラム: latitude, longitude, label）"""
    rows = [{"latitude": s[0], "longitude": s[1], "label": s[2]} for s in data if len(s) >= 3]
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "label"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ----------------------------------------
# 音圧計算とグリッド生成
# ----------------------------------------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各グリッド点で全スピーカーからの音圧(dB)を合算する。
    単純な距離減衰モデル: L0 - 20 log10(distance)
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, _ = spk
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

def generate_grid(center, delta=0.01, resolution=60):
    """中心座標とdeltaからグリッド (grid_lat, grid_lon) を生成"""
    lat, lon = center
    lat_min, lat_max = lat - delta, lat + delta
    lon_min, lon_max = lon - delta, lon + delta
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, resolution),
        np.linspace(lon_min, lon_max, resolution)
    )
    # 転置して shape を (resolution, resolution) に
    return grid_lat.T, grid_lon.T

def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    """音圧分布データ(DataFrame)を生成。カラム: latitude, longitude, weight"""
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                data.append({"latitude": grid_lat[i, j], "longitude": grid_lon[i, j], "weight": val})
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

def get_column_data(grid_lat, grid_lon, speakers, L0, r_max):
    """3D ColumnLayer 用に、各グリッド点の音圧値、正規化・高さ、色データをDataFrame化する"""
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data_list = []
    try:
        val_min = np.nanmin(sound_grid)
        val_max = np.nanmax(sound_grid)
    except Exception:
        return pd.DataFrame()
    if math.isnan(val_min) or val_min == val_max:
        return pd.DataFrame()
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                norm = (val - val_min) / (val_max - val_min)
                elevation = norm * 1000.0  # 調整可能な高さスケール
                # 色: norm=0 -> 青, norm=1 -> 赤 (Gは逆)
                r = int(255 * norm)
                g = int(255 * (1 - norm))
                b = 128
                data_list.append({
                    "lat": grid_lat[i, j],
                    "lon": grid_lon[i, j],
                    "value": val,
                    "elevation": elevation,
                    "color": [r, g, b]
                })
    return pd.DataFrame(data_list)

# ----------------------------------------
# Gemini API 連携関連
# ----------------------------------------
def generate_gemini_prompt(user_query):
    """Gemini へのプロンプトを生成。座標表記は「緯度: 34.255500, 経度: 133.207000」で統一するよう指示"""
    speakers = st.session_state.speakers if "speakers" in st.session_state else []
    if speakers:
        speaker_info = "\n".join(
            f"{i+1}. 緯度: {s[0]:.6f}, 経度: {s[1]:.6f}, ラベル: {s[2]}"
            for i, s in enumerate(speakers)
        )
    else:
        speaker_info = "現在、スピーカーは配置されていません。"
    sound_range = f"{st.session_state.L0 - 40}dB ~ {st.session_state.L0}dB"
    prompt = (
        f"配置されているスピーカー:\n{speaker_info}\n"
        f"現在の音圧レベルの範囲: {sound_range}\n"
        "海など設置困難な場所は除外し、スピーカー同士は300m以上離れている場所を考慮してください。\n"
        f"ユーザーの問い合わせ: {user_query}\n"
        "上記情報に基づき、改善案を具体的かつ詳細に提案してください。\n"
        "【座標表記形式】 緯度: 34.255500, 経度: 133.207000 で統一してください。"
    )
    return prompt

def call_gemini_api(query):
    """Gemini API へクエリを送信して回答テキストを返す"""
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

def extract_speaker_proposals(response_text):
    """
    Gemini の回答からスピーカー提案（新規）を抽出する。
    戻り値は [[lat, lon, label, "new"], ...] の形式。
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
    """Gemini の回答に含まれる新規スピーカーを抽出して追加する"""
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

# ----------------------------------------
# キャッシュ設定とスレッドプール
# ----------------------------------------
executor = ThreadPoolExecutor(max_workers=2)

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

# ----------------------------------------
# Pydeck レイヤー作成
# ----------------------------------------
def create_scatter_layer(spk_df):
    """スピーカーをScatterplotLayerで表示。新規は赤色、既存は青色"""
    return pdk.Layer(
        "ScatterplotLayer",
        data=spk_df,
        get_position=["lon", "lat"],
        get_radius=100,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

def create_heatmap_layer(heat_df):
    """HeatmapLayer の作成。透明度を高めに設定"""
    return pdk.Layer(
        "HeatmapLayer",
        data=heat_df,
        get_position=["longitude", "latitude"],
        get_weight="weight",
        radiusPixels=50,
        min_opacity=0.1,
        max_opacity=0.3,
    )

def create_column_layer(col_df):
    """3Dカラム表示用の ColumnLayer の作成"""
    return pdk.Layer(
        "ColumnLayer",
        data=col_df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=30,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

# ----------------------------------------
# Main UI
# ----------------------------------------
def main():
    st.title("防災スピーカー音圧可視化マップ")
    
    # セッション初期化（中心、ズーム、スピーカー、パラメータ）
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

    # ---------- サイドバー ----------
    with st.sidebar:
        st.header("操作パネル")
        # CSVアップロード
        uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
        if uploaded_file and st.button("CSVからスピーカー登録"):
            new_speakers = load_csv(uploaded_file)
            if new_speakers:
                st.session_state.speakers.extend(new_speakers)
                st.session_state.heatmap_data = None
                st.success(f"CSVから {len(new_speakers)} 件のスピーカー情報を追加しました。")
            else:
                st.error("CSVに正しいデータが見つかりませんでした。")
        # 手動追加
        new_speaker = st.text_input("スピーカー追加 (latitude,longitude,label)", placeholder="例: 34.2579,133.2072,役場")
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
        # 削除・編集
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
        if st.session_state.get("edit_index") is not None:
            with st.form("edit_form"):
                spk = st.session_state.speakers[st.session_state.edit_index]
                new_lat = st.text_input("新しい緯度", value=str(spk[0]))
                new_lon = st.text_input("新しい経度", value=str(spk[1]))
                new_label = st.text_input("新しいラベル", value=spk[2])
                if st.form_submit_button("編集保存"):
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
        # パラメータ調整
        st.session_state.L0 = st.slider("初期音圧レベル (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
        # 表示モード選択: "HeatMap" または "3D Columns"
        display_mode = st.radio("表示モードを選択", ["HeatMap", "3D Columns"])
        # Gemini API
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_input("Gemini に問い合わせる内容")
        if st.button("Gemini API を実行"):
            prompt = generate_gemini_prompt(gemini_query)
            result = call_gemini_api(prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API 実行完了")
            add_speaker_proposals_from_gemini()
    
    # ---------- グリッド生成 ----------
    grid_lat, grid_lon = generate_grid(st.session_state.map_center, delta=0.01, resolution=60)
    
    # ---------- Pydeck 用スピーカー DataFrame ----------
    spk_list = []
    for s in st.session_state.speakers:
        flag = s[3] if len(s) >= 4 else ""
        spk_list.append([s[0], s[1], s[2], flag])
    spk_df = pd.DataFrame(spk_list, columns=["lat", "lon", "label", "flag"])
    spk_df["color"] = spk_df["flag"].apply(lambda x: [255, 0, 0] if x == "new" else [0, 0, 255])
    
    layers = []
    if display_mode == "HeatMap":
        if st.session_state.heatmap_data is None:
            st.session_state.heatmap_data = cached_calculate_heatmap(
                st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
            )
        if not st.session_state.heatmap_data.empty:
            heatmap_layer = create_heatmap_layer(st.session_state.heatmap_data)
            layers.append(heatmap_layer)
        else:
            st.info("ヒートマップデータが空です。")
        scatter_layer = create_scatter_layer(spk_df)
        layers.append(scatter_layer)
    else:
        col_df = get_column_data(grid_lat, grid_lon, st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
        if col_df.empty:
            st.info("3Dカラム用データが空です。")
        else:
            column_layer = create_column_layer(col_df)
            layers.append(column_layer)
        scatter_layer = create_scatter_layer(spk_df)
        layers.append(scatter_layer)
    
    # ---------- 全体音圧範囲の計算表示 ----------
    try:
        sound_grid = compute_sound_grid(st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon)
        dB_min = np.nanmin(sound_grid)
        dB_max = np.nanmax(sound_grid)
        st.write(f"全スピーカーの音圧範囲: {dB_min:.1f} dB ～ {dB_max:.1f} dB")
    except Exception:
        st.write("音圧範囲の計算に失敗しました。")
    
    if layers:
        view_state = pdk.ViewState(
            latitude=st.session_state.map_center[0],
            longitude=st.session_state.map_center[1],
            zoom=st.session_state.map_zoom,
            pitch=45 if display_mode=="3D Columns" else 0,
            bearing=0,
        )
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{label}\n音圧: {value}"}
        )
        st.pydeck_chart(deck)
    else:
        st.write("表示するレイヤーがありません。")
    
    csv_data = export_csv(st.session_state.speakers)
    st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        if st.session_state.heatmap_data is not None:
            st.write("ヒートマップデータ件数:", len(st.session_state.heatmap_data))
        st.write("表示モード:", display_mode)
        st.write("L0:", st.session_state.L0, "r_max:", st.session_state.r_max)
    
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
