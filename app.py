import os
import math
import io
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

# ------------------ 初期設定 ------------------
st.set_page_config(page_title="上島町　防災無線スピーカ設置場所考えるくん", layout="wide")

CUSTOM_CSS = """
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
    text-align: left;
}
div.stButton > button:hover { background-color: #45a049; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-weight: bold; 
    color: #4CAF50;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------ API設定 ------------------
API_KEY = st.secrets["general"]["api_key"]  # secrets.toml の [general] に設定
MODEL_NAME = "gemini-2.0-flash"

# ------------------ 方向文字列 → 度数変換 ------------------
DIRECTION_MAP = {
    "N": 0, "E": 90, "S": 180, "W": 270,
    "NE": 45, "SE": 135, "SW": 225, "NW": 315
}

def parse_direction(dir_str):
    dir_str = dir_str.strip().upper()
    if dir_str in DIRECTION_MAP:
        return float(DIRECTION_MAP[dir_str])
    try:
        return float(dir_str)
    except:
        st.warning(f"方向 '{dir_str}' は不正です。0度とみなします。")
        return 0.0

# ------------------ CSV 読み込み／書き出し ------------------
def load_csv(file):
    """CSVからスピーカー情報を読み込み、[[lat, lon, label, direction], ...] を返す。"""
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
                if "direction" in df.columns and not pd.isna(row.get("direction")):
                    direction = parse_direction(str(row["direction"]))
                else:
                    direction = 0.0
                speakers.append([lat, lon, label, direction])
            except Exception as e:
                st.warning(f"行 {idx+1} 読み込み失敗: {e}")
        return speakers
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return []

def export_csv(data):
    """スピーカー情報をCSV形式に変換（directionは出力しない）"""
    rows = []
    for s in data:
        lat, lon, label = s[0], s[1], s[2]
        rows.append({"latitude": lat, "longitude": lon, "label": label})
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "label"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ------------------ 音圧計算（方向対応） ------------------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各スピーカーからの音圧 (dB) を計算する。
    方向情報により、グリッド点への角度差のコサイン補正（最低0.3倍）を掛ける。
    結果は (L0-40) ～ L0 にクリップされる。
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat_s, lon_s, label = spk[0], spk[1], spk[2]
        direction_deg = float(spk[3]) if len(spk) >= 4 else 0.0
        dlat = grid_lat - lat_s
        dlon = grid_lon - lon_s
        distance = np.sqrt((dlat * 111320)**2 + (dlon * 111320 * math.cos(math.radians(lat_s)))**2)
        distance[distance < 1] = 1
        p_db = L0 - 20 * np.log10(distance)
        # 方向依存補正
        bearing = (np.degrees(np.arctan2(dlon, dlat))) % 360
        angle_diff = np.abs(bearing - direction_deg)
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        directional_factor = np.cos(np.radians(angle_diff))
        directional_factor[directional_factor < 0.3] = 0.3
        p_linear = 10 ** (p_db / 10)
        power = directional_factor * p_linear
        power[distance > r_max] = 0
        power_sum += power
    total_db = np.full_like(power_sum, np.nan)
    mask = power_sum > 0
    total_db[mask] = 10 * np.log10(power_sum[mask])
    total_db = np.clip(total_db, L0 - 40, L0)
    return total_db

# ------------------ グリッド生成（上島町全域） ------------------
def generate_grid_for_kamijima(resolution=80):
    lat_min, lat_max = 34.20, 34.35
    lon_min, lon_max = 133.15, 133.28
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, resolution),
        np.linspace(lon_min, lon_max, resolution)
    )
    return grid_lat.T, grid_lon.T

def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    sgrid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data = []
    for i in range(Nx):
        for j in range(Ny):
            data.append({"latitude": grid_lat[i, j], "longitude": grid_lon[i, j], "weight": sgrid[i, j]})
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

def get_column_data(grid_lat, grid_lon, speakers, L0, r_max):
    """
    ColumnLayer用データ生成：
    音圧を正規化し、高さを 0～500 に、色は弱→青、強→赤（半透明）に設定。
    """
    sgrid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    if np.all(np.isnan(sgrid)):
        return pd.DataFrame()
    Nx, Ny = grid_lat.shape
    data_list = []
    val_min = np.nanmin(sgrid)
    val_max = np.nanmax(sgrid)
    if math.isnan(val_min) or val_min == val_max:
        return pd.DataFrame()
    for i in range(Nx):
        for j in range(Ny):
            val = sgrid[i, j]
            if not np.isnan(val):
                norm = (val - val_min) / (val_max - val_min)
                elevation = norm * 500.0  # 高さを0～500に拡大
                r = int(255 * norm)
                g = int(255 * (1 - norm))
                b = 128
                a = 120
                data_list.append({
                    "lat": grid_lat[i, j],
                    "lon": grid_lon[i, j],
                    "value": val,
                    "elevation": elevation,
                    "color": [r, g, b, a]
                })
    return pd.DataFrame(data_list)

# ------------------ 円形ジオJSON生成 ------------------
def create_circle_geojson(lat, lon, radius, num_points=50):
    """
    指定した中心 (lat, lon) と半径 (m) で、円形のGeoJSONを生成する。
    """
    points = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        dlat = (radius * math.sin(angle)) / 111320
        dlon = (radius * math.cos(angle)) / (111320 * math.cos(math.radians(lat)))
        points.append([lon + dlon, lat + dlat])
    points.append(points[0])
    geojson = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [points]},
        "properties": {}
    }
    return geojson

# ------------------ ヒートマップの動的設定 ------------------
def dynamic_radius_pixels(zoom):
    """ズームレベルに応じて、radiusPixels を動的に調整する（zoom=11 のときは約15を基準とする）"""
    return max(200, int(50 * (30 / zoom)))

def create_heatmap_layer(heat_df):
    return pdk.Layer(
        "HeatmapLayer",
        data=heat_df,
        get_position=["longitude", "latitude"],
        get_weight="weight",
        radiusPixels=dynamic_radius_pixels(st.session_state.map_zoom),
        min_opacity=0.1,   # 透明度の下限（より透明に）
        max_opacity=0.3,   # 透明度の上限（背景が見やすい）
         colorRange=[
            [0, 0, 255, 150],     # 青（低音圧、半透明）
            [0, 255, 255, 150],   # シアン
            [0, 255, 0, 150],     # 緑
            [255, 255, 0, 150],   # 黄色
            [255, 0, 0, 150]      # 赤（高音圧、半透明）
        ]
    )

def create_column_layer(col_df):
    return pdk.Layer(
        "ColumnLayer",
        data=col_df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=20,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

# ------------------ Gemini API 連携 ------------------
def generate_gemini_prompt(user_query):
    spk_info = ""
    if st.session_state.speakers:
        spk_info = "\n".join(
            f"{i+1}. 緯度:{s[0]:.6f}, 経度:{s[1]:.6f}, ラベル:{s[2]}, 方向:{s[3] if len(s)>=4 else 0}"
            for i, s in enumerate(st.session_state.speakers)
        )
    else:
        spk_info = "現在、スピーカーは配置されていません。"
    sound_range = f"{st.session_state.L0-40}dB ~ {st.session_state.L0}dB"
    prompt = (
        f"配置されているスピーカー:\n{spk_info}\n"
        f"現在の音圧レベルの範囲: {sound_range}\n"
        "海など設置困難な場所は除外し、スピーカー同士は300m以上離れているよう考慮してください。\n"
        f"ユーザー問い合わせ: {user_query}\n"
        "上記情報に基づき、改善案を具体的かつ詳細に提案してください。\n"
        "【座標表記】 緯度:34.255500, 経度:133.207000 で統一。"
    )
    return prompt

def call_gemini_api(query):
    headers = {"Content-Type": "application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload = {"contents": [{"parts": [{"text": query}]}]}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        rjson = resp.json()
        cands = rjson.get("candidates", [])
        if not cands:
            st.error("Gemini API エラー: candidatesが空")
            return "回答が得られませんでした。"
        c0 = cands[0]
        cv = c0.get("content", "")
        if isinstance(cv, dict):
            parts = cv.get("parts", [])
            text = " ".join([p.get("text", "") for p in parts])
        else:
            text = str(cv)
        return text.strip()
    except Exception as e:
        st.error(f"Gemini API呼び出しエラー: {e}")
        return f"エラー: {e}"

def extract_speaker_proposals(res_text):
    pattern = r"(?:緯度[:：]?\s*)([-\d]+\.\d+)[,、\s]+(?:経度[:：]?\s*)([-\d]+\.\d+)(?:[,、\s]+(?:方向[:：]?\s*([^\n,]+))?)?"
    props = re.findall(pattern, res_text)
    results = []
    for lat_str, lon_str, dir_str in props:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            direction_deg = parse_direction(dir_str) if dir_str else 0.0
            results.append([lat, lon, "Gemini提案", direction_deg])
        except:
            continue
    return results

def add_speaker_proposals_from_gemini():
    if not st.session_state.get("gemini_result"):
        st.error("Gemini API の回答がありません。")
        return
    proposals = extract_speaker_proposals(st.session_state["gemini_result"])
    if proposals:
        added = 0
        for p in proposals:
            if not any(abs(p[0]-s[0]) < 1e-6 and abs(p[1]-s[1]) < 1e-6 for s in st.session_state.speakers):
                st.session_state.speakers.append(p)
                added += 1
        if added > 0:
            st.success(f"{added}件のGemini提案スピーカーを追加しました。")
            st.session_state.heatmap_data = None
        else:
            st.info("新規スピーカーは見つかりませんでした。")
    else:
        st.info("Gemini回答にスピーカー情報が見つかりません。")

# ------------------ ScenegraphLayer (初期スピーカーの3Dモデル) ------------------
def create_speaker_3d_layer(spk_df):
    SCENEGRAPH_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/scenegraph/airplane/scene.gltf"
    spk_df["z"] = 50  # 初期スピーカーは3Dモデルで表示（高さ50に固定）
    return pdk.Layer(
        "ScenegraphLayer",
        data=spk_df,
        scenegraph=SCENEGRAPH_URL,
        get_position=["lon", "lat", "z"],
        get_orientation=[0, 0, 0],
        sizeScale=20,
        pickable=True
    )

# ------------------ 追加スピーカー用 小さい散布図レイヤー ------------------
def create_small_scatter_layer(spk_df):
    return pdk.Layer(
        "ScatterplotLayer",
        data=spk_df,
        get_position=["lon", "lat"],
        get_radius=30,  # 小さい円
        get_fill_color="[0, 0, 255, 255]",  # 青色
        pickable=True,
        auto_highlight=True,
    )

# ------------------ Pydeck レイヤー作成 ------------------
def create_column_layer(col_df):
    return pdk.Layer(
        "ColumnLayer",
        data=col_df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,
        radius=20,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

# ------------------ 全スピーカー伝搬アニメーション用 ------------------
def animate_all_propagation(speakers, base_layers, view_state, L0, r_max, repeat=2):
    """
    全スピーカーからの音圧伝搬アニメーションを、指定回数 (repeat) 繰り返し表示する。
    最大伝播距離 (r_max) に応じた円の最大半径でアニメーションを行い、
    最終サイクルではフェードアウトせず、円の状態をそのまま維持する。
    """
    container = st.empty()
    num_steps = 20
    max_radius = r_max  # 最大伝播距離を利用
    final_deck = None
    for cycle in range(repeat):
        # 拡大フェーズ
        for step in range(1, num_steps + 1):
            radius = (step / num_steps) * max_radius
            anim_layers = []
            for spk in speakers:
                circle_geo = create_circle_geojson(spk[0], spk[1], radius)
                effective_radius = radius if radius >= 1 else 1
                p_db = L0 - 20 * math.log10(effective_radius)
                alpha = int(255 * (p_db - (L0 - 40)) / 40)
                alpha = max(0, min(alpha, 255))
                # 円の塗りと線を青に設定
                anim_layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=circle_geo,
                    get_fill_color=[0, 0, 255, alpha],
                    get_line_color=[0, 0, 255],
                    line_width_min_pixels=2,
                )
                anim_layers.append(anim_layer)
            current_layers = base_layers + anim_layers
            deck = pdk.Deck(
                layers=current_layers,
                initial_view_state=view_state,
                tooltip={"text": "{label}\n方向: {direction} d\n音圧: {value} dB"}
            )
            container.pydeck_chart(deck)
            time.sleep(0.25)
            final_deck = deck
        # 最終サイクルではフェードアウトせずそのまま状態を維持
        if cycle < repeat - 1:
            for fade in range(10, -1, -1):
                fade_alpha = int(80 * (fade / 10))
                anim_layers = []
                for spk in speakers:
                    circle_geo = create_circle_geojson(spk[0], spk[1], max_radius)
                    anim_layer = pdk.Layer(
                        "GeoJsonLayer",
                        data=circle_geo,
                        get_fill_color=[0, 0, 255, fade_alpha],
                        get_line_color=[0, 0, 255],
                        line_width_min_pixels=2,
                    )
                    anim_layers.append(anim_layer)
                current_layers = base_layers + anim_layers
                deck = pdk.Deck(
                    layers=current_layers,
                    initial_view_state=view_state,
                    tooltip={"text": "{label}\n方向: {direction} d\n音圧: {value} dB"}
                )
                container.pydeck_chart(deck)
                time.sleep(0.15)
                final_deck = deck
    # アニメーション終了後、container.empty() は呼ばず最終状態を維持
    return final_deck

# ------------------ 個別スピーカー伝搬アニメーション用 ------------------
def animate_individual_propagation(speaker, base_layers, view_state, L0, r_max, repeat=2):
    return animate_all_propagation([speaker], base_layers, view_state, L0, r_max, repeat=repeat)

# ------------------ メインUI ------------------
def main():
    st.title("上島町　防災無線スピーカ設置場所考えるくん")
    
    # セッション初期化
    if "map_center" not in st.session_state:
        st.session_state.map_center = [34.25, 133.20]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 11
    if "speakers" not in st.session_state:
        st.session_state.speakers = [[34.25, 133.20, "初期スピーカーA", 0.0]]
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
    if "animate_all" not in st.session_state:
        st.session_state.animate_all = False
    if "animate_individual" not in st.session_state:
        st.session_state.animate_individual = False
    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None

    # サイドバー操作
    with st.sidebar:
        st.header("操作パネル")
        upfile = st.file_uploader("CSVアップロード", type=["csv"])
        if upfile and st.button("CSVからスピーカー登録"):
            new_spk = load_csv(upfile)
            if new_spk:
                st.session_state.speakers.extend(new_spk)
                st.session_state.heatmap_data = None
                st.success(f"{len(new_spk)}件追加")
            else:
                st.error("CSVに有効データなし")
        
        new_text = st.text_input("スピーカー追加 (lat,lon,label,方向)", placeholder="例: 34.25,133.20,役場,N")
        if st.button("スピーカー追加"):
            parts = new_text.split(",")
            if len(parts) < 2:
                st.error("緯度,経度は必要です")
            else:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    label = parts[2].strip() if len(parts) > 2 else "新スピーカー"
                    dir_str = parts[3].strip() if len(parts) > 3 else "0"
                    direction_deg = parse_direction(dir_str)
                    st.session_state.speakers.append([lat, lon, label, direction_deg])
                    st.session_state.heatmap_data = None
                    st.success(f"追加成功: {lat}, {lon}, {label}, 方向: {direction_deg}")
                except Exception as e:
                    st.error(f"追加失敗: {e}")
        
        if st.session_state.speakers:
            opts = [f"{i}: ({s[0]:.4f},{s[1]:.4f}) - {s[2]}, 方向:{s[3] if len(s)>=4 else 0}" 
                    for i, s in enumerate(st.session_state.speakers)]
            sel = st.selectbox("スピーカー選択", list(range(len(opts))), format_func=lambda i: opts[i])
            st.session_state.selected_index = sel
            c1, c2 = st.columns(2)
            with c1:
                if st.button("選択削除"):
                    try:
                        del st.session_state.speakers[sel]
                        st.session_state.heatmap_data = None
                        st.success("削除成功")
                    except Exception as e:
                        st.error(f"削除失敗: {e}")
            with c2:
                if st.button("選択編集"):
                    st.session_state.edit_index = sel
        else:
            st.info("スピーカーがありません。")
        
        if st.session_state.get("edit_index") is not None:
            with st.form("edit_form"):
                spk = st.session_state.speakers[st.session_state.edit_index]
                new_lat = st.text_input("緯度", value=str(spk[0]))
                new_lon = st.text_input("経度", value=str(spk[1]))
                new_lbl = st.text_input("ラベル", value=spk[2])
                new_dir = st.text_input("方向", value=str(spk[3] if len(spk)>=4 else "0"))
                if st.form_submit_button("編集保存"):
                    try:
                        latv = float(new_lat)
                        lonv = float(new_lon)
                        lblv = new_lbl
                        dir_deg = parse_direction(new_dir)
                        st.session_state.speakers[st.session_state.edit_index] = [latv, lonv, lblv, dir_deg]
                        st.session_state.heatmap_data = None
                        st.success("編集成功")
                        st.session_state.edit_index = None
                    except Exception as e:
                        st.error(f"編集失敗: {e}")
        
        if st.button("スピーカーリセット"):
            st.session_state.speakers = []
            st.session_state.heatmap_data = None
            st.success("リセット完了")
        
        st.session_state.L0 = st.slider("初期音圧 (dB)", 50, 100, st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離 (m)", 100, 2000, st.session_state.r_max)
        disp_mode = st.radio("表示モード", ["HeatMap", "3D Columns"])
        
        st.subheader("Gemini API")
        gem_query = st.text_input("問い合わせ内容")
        if st.button("Gemini API 実行"):
            prompt = generate_gemini_prompt(gem_query)
            ans = call_gemini_api(prompt)
            st.session_state.gemini_result = ans
            st.success("Gemini完了")
            add_speaker_proposals_from_gemini()
        
        # 個別と全体の伝搬アニメーションボタン
        if st.session_state.speakers:
            if st.button("個別スピーカー伝搬アニメーション表示"):
                st.session_state.animate_individual = True
            if st.button("全スピーカー伝搬アニメーション表示"):
                st.session_state.animate_all = True

    # ------------------ グリッド生成 ------------------
    grid_lat, grid_lon = generate_grid_for_kamijima(resolution=80)
    
    # ------------------ スピーカーのデータ分割 ------------------
    # 初期スピーカー（最初の1件）は3Dモデル表示、追加スピーカーは小さい青丸で表示
    default_spk_list = st.session_state.speakers[0:1] if len(st.session_state.speakers) >= 1 else []
    added_spk_list = st.session_state.speakers[1:] if len(st.session_state.speakers) > 1 else []
    default_spk_df = pd.DataFrame(default_spk_list, columns=["lat", "lon", "label", "direction"]) if default_spk_list else pd.DataFrame(columns=["lat", "lon", "label", "direction"])
    if not default_spk_df.empty:
        default_spk_df["z"] = 50
    added_spk_df = pd.DataFrame(added_spk_list, columns=["lat", "lon", "label", "direction"]) if added_spk_list else pd.DataFrame(columns=["lat", "lon", "label", "direction"])
    
    # ------------------ レイヤー作成 ------------------
    layers = []
    if disp_mode == "HeatMap":
        if st.session_state.heatmap_data is None:
            st.session_state.heatmap_data = cached_calculate_heatmap(
                st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
            )
        if not st.session_state.heatmap_data.empty:
            heatmap_layer = create_heatmap_layer(st.session_state.heatmap_data)
            layers.append(heatmap_layer)
        else:
            st.info("ヒートマップデータが空です。")
    else:
        col_df = get_column_data(grid_lat, grid_lon, st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
        if not col_df.empty:
            column_layer = create_column_layer(col_df)
            layers.append(column_layer)
        else:
            st.info("3Dカラムデータが空です。")
    
    # 初期スピーカーは3Dモデル、追加スピーカーは小さい青丸で表示
    default_layer = create_speaker_3d_layer(default_spk_df) if not default_spk_df.empty else None
    added_layer = create_small_scatter_layer(added_spk_df) if not added_spk_df.empty else None
    if default_layer:
        layers.append(default_layer)
    if added_layer:
        layers.append(added_layer)
    
    # ------------------ 全体音圧範囲の表示 ------------------
    sgrid = compute_sound_grid(st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon)
    try:
        dbmin = np.nanmin(sgrid)
        dbmax = np.nanmax(sgrid)
        st.write(f"全スピーカーの音圧範囲 (上島町全域): {dbmin:.1f} dB ～ {dbmax:.1f} dB")
    except:
        st.write("音圧範囲計算失敗")
    
    # ------------------ ビュー設定 ------------------
    view_state = pdk.ViewState(
        latitude=34.25,
        longitude=133.20,
        zoom=st.session_state.map_zoom,
        pitch=45 if disp_mode=="3D Columns" else 0,
        bearing=0
    )
    base_deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "{label}\n方向: {direction} d\n音圧: {value} dB"}
    )
    
    # ------------------ アニメーション処理 ------------------
    if st.session_state.get("animate_all", False):
        final_deck = animate_all_propagation(
            st.session_state.speakers, layers.copy(), view_state, st.session_state.L0, st.session_state.r_max, repeat=2
        )
        st.session_state.animate_all = False
        st.pydeck_chart(final_deck)
    elif st.session_state.get("animate_individual", False) and st.session_state.selected_index is not None:
        selected_spk = st.session_state.speakers[st.session_state.selected_index]
        final_deck = animate_individual_propagation(
            selected_spk, layers.copy(), view_state, st.session_state.L0, st.session_state.r_max, repeat=2
        )
        st.session_state.animate_individual = False
        st.pydeck_chart(final_deck)
    else:
        st.pydeck_chart(base_deck)
    
    # ------------------ CSV ダウンロード ------------------
    csv_data = export_csv(st.session_state.speakers)
    st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv", "text/csv")
    
    with st.expander("デバッグ情報"):
        st.write("スピーカー:", st.session_state.speakers)
        st.write("表示モード:", disp_mode)
        st.write("L0:", st.session_state.L0, "r_max:", st.session_state.r_max)
    
    st.markdown("---")
    st.subheader("Gemini API の回答（テキスト表示）")
    if st.session_state.gemini_result:
        st.text(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答はありません。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラー: {e}")
