import os
import math
import io
import re
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
import pydeck as pdk

# ----------------- 初期設定 -----------------
st.set_page_config(page_title="愛媛県上島町 全域 + スピーカー方向対応", layout="wide")

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

# ----------------- API設定 -----------------
API_KEY = st.secrets["general"]["api_key"]  # secrets.toml の [general] に設定
MODEL_NAME = "gemini-2.0-flash"

# ----------------- 方向文字列 → 度数変換 -----------------
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

# ----------------- CSV 読み込み／書き出し -----------------
def load_csv(file):
    """CSVからスピーカー情報を読み込み、[[lat, lon, label, direction_deg], ...] を返す。
       CSVに方向カラムがある場合は"direction"カラムを使用。"""
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
                # 方向カラムが存在すればそれを使う（ない場合はデフォルト0°）
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
    """スピーカー情報をCSVに変換（directionカラムは出力しない）"""
    rows = []
    for s in data:
        # sは [lat, lon, label, (optionally direction)]
        lat, lon, label = s[0], s[1], s[2]
        rows.append({"latitude": lat, "longitude": lon, "label": label})
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "label"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ----------------- 音圧計算（方向対応） -----------------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各スピーカーの音圧を計算する。
    各スピーカーについて、方向が与えられている場合はその方向に対するコサイン減衰を掛ける（最低0.3倍）。
    最終的に全スピーカーの寄与を合算し、dB 値に変換する。
    結果は [L0-40, L0] の範囲にクリップされる。
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
        # 方向依存：グリッド点への方位
        bearing = (np.degrees(np.arctan2(dlon, dlat))) % 360
        angle_diff = np.abs(bearing - direction_deg) % 360
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

# ----------------- グリッド生成（上島町全域） -----------------
def generate_grid_for_kamijima(resolution=80):
    """
    愛媛県上島町全域をカバーするグリッドを生成する。
      緯度: 34.20 ~ 34.35
      経度: 133.15 ~ 133.28
    """
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
    sgrid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data_list = []
    val_min = np.nanmin(sgrid)
    val_max = np.nanmax(sgrid)
    if math.isnan(val_min) or val_min == val_max:
        return pd.DataFrame()
    for i in range(Nx):
        for j in range(Ny):
            val = sgrid[i, j]
            norm = (val - val_min) / (val_max - val_min)
            # 高さを抑える: 0～300 の範囲
            elevation = norm * 300.0
            # 色のグラデーション: norm=0 -> 青, norm=1 -> 赤。α=120 で半透明。
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

# ----------------- Gemini API 関連 -----------------
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
        "海など設置困難な場所は除外し、スピーカー同士は300m以上離れている場所を考慮してください。\n"
        f"ユーザーの問い合わせ: {user_query}\n"
        "上記情報に基づき、改善案を具体的かつ詳細に提案してください。\n"
        "【座標表記形式】 緯度:34.255500, 経度:133.207000 で統一してください。"
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

def extract_speaker_proposals(response_text):
    # 対応する形式例: "緯度:34.259000, 経度:133.201500, 方向:N" など
    pattern = r"(?:緯度[:：]?\s*)([-\d]+\.\d+)[,、\s]+(?:経度[:：]?\s*)([-\d]+\.\d+)(?:[,、\s]+(?:方向[:：]?\s*([^\n,]+))?)?"
    props = re.findall(pattern, response_text)
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
            if not any(abs(p[0]-s[0])<1e-6 and abs(p[1]-s[1])<1e-6 for s in st.session_state.speakers):
                st.session_state.speakers.append(p)
                added += 1
        if added > 0:
            st.success(f"{added}件のGemini提案スピーカーを追加しました。")
            st.session_state.heatmap_data = None
        else:
            st.info("新規スピーカーは見つかりませんでした。")
    else:
        st.info("Gemini回答にスピーカー情報が見つかりません。")

# ----------------- ScenegraphLayer 用 3D スピーカー -----------------
def create_speaker_3d_layer(spk_df):
    # 3Dモデル URL（例: 飛行機モデル）
    SCENEGRAPH_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/scenegraph/airplane/scene.gltf"
    spk_df["z"] = 30  # スピーカー表示用の高さ
    return pdk.Layer(
        "ScenegraphLayer",
        data=spk_df,
        scenegraph=SCENEGRAPH_URL,
        get_position=["lon", "lat", "z"],
        get_orientation=[0,0,0],
        sizeScale=20,
        pickable=True
    )

# ----------------- Pydeck レイヤー作成 -----------------
def create_scatter_layer(spk_df):
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

# ----------------- メインUI -----------------
def main():
    st.title("愛媛県上島町 全域＋スピーカー方向対応")
    
    # セッション初期化
    if "map_center" not in st.session_state:
        st.session_state.map_center = [34.25, 133.20]  # 上島町中心
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 11
    if "speakers" not in st.session_state:
        st.session_state.speakers = [
            [34.25, 133.20, "初期スピーカーA", 0.0]
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
                    st.error(f"追加エラー: {e}")
        
        if st.session_state.speakers:
            opts = [f"{i}: ({s[0]:.4f},{s[1]:.4f}) - {s[2]}, 方向:{s[3] if len(s)>=4 else 0}" for i, s in enumerate(st.session_state.speakers)]
            sel = st.selectbox("スピーカー選択", list(range(len(opts))), format_func=lambda i: opts[i])
            col1, col2 = st.columns(2)
            with col1:
                if st.button("選択削除"):
                    try:
                        del st.session_state.speakers[sel]
                        st.session_state.heatmap_data = None
                        st.success("削除成功")
                    except Exception as e:
                        st.error(f"削除失敗: {e}")
            with col2:
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
                new_dir = st.text_input("方向", value=str(spk[3] if len(spk) >= 4 else "0"))
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
    
    # グリッド生成：上島町全域 (34.20〜34.35, 133.15〜133.28)
    grid_lat, grid_lon = generate_grid_for_kamijima(resolution=80)
    
    # スピーカー DataFrame作成
    spk_list = []
    for s in st.session_state.speakers:
        # sは [lat, lon, label, direction_deg] もしくは [lat, lon, label]
        flag = s[3] if len(s) >= 4 else ""
        spk_list.append([s[0], s[1], s[2], flag])
    spk_df = pd.DataFrame(spk_list, columns=["lat", "lon", "label", "flag"])
    # Scenegraph用 z 座標
    spk_df["z"] = 30
    spk_df["color"] = spk_df["flag"].apply(lambda x: [255, 0, 0] if x=="new" else [0, 0, 255])
    
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
        scatter_layer = create_scatter_layer(spk_df)
        layers.append(scatter_layer)
    else:
        # 3D Columnsモード
        col_df = get_column_data(grid_lat, grid_lon, st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
        if not col_df.empty:
            column_layer = create_column_layer(col_df)
            layers.append(column_layer)
        else:
            st.info("3Dカラムデータが空です。")
        scatter_layer = create_scatter_layer(spk_df)
        layers.append(scatter_layer)
    
    # スピーカー 3Dモデル（ScenegraphLayer）を後から追加 => 上に表示
    speaker_3d_layer = create_speaker_3d_layer(spk_df)
    layers.append(speaker_3d_layer)
    
    # 全体音圧範囲の計算表示
    sgrid = compute_sound_grid(st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon)
    try:
        dbmin = np.nanmin(sgrid)
        dbmax = np.nanmax(sgrid)
        st.write(f"全スピーカーの音圧範囲 (上島町全域): {dbmin:.1f} dB ～ {dbmax:.1f} dB")
    except:
        st.write("音圧範囲計算失敗")
    
    if layers:
        view_state = pdk.ViewState(
            latitude=34.25,
            longitude=133.20,
            zoom=st.session_state.map_zoom,
            pitch=45 if disp_mode=="3D Columns" else 0,
            bearing=0
        )
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{label}\n方向: {flag}\n音圧: {value}"}
        )
        st.pydeck_chart(deck)
    else:
        st.write("表示レイヤーなし")
    
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
