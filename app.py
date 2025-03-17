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
import matplotlib.pyplot as plt

st.set_page_config(page_title="防災スピーカー音圧可視化マップ", layout="wide")

# ---------- CSV読み込み・書き出し ----------
def load_csv(file):
    """
    CSVファイルを読み込み、[[lat, lon, label], ...] 形式のスピーカー情報を返す。
    'latitude', 'longitude' カラムが必須。
    'label', '施設名', '名称' のいずれかがあればラベルとして利用。
    """
    try:
        df = pd.read_csv(file)
        speakers = []
        for idx, row in df.iterrows():
            try:
                lat_val = row.get("latitude")
                lon_val = row.get("longitude")
                if pd.isna(lat_val) or pd.isna(lon_val):
                    st.warning(f"行 {idx+1} に欠損した座標があるためスキップ")
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
                st.warning(f"行 {idx+1} の読み込み失敗: {e}")
        return speakers
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return []

def export_csv(data):
    """
    [[lat, lon, label], ...] or [[lat, lon, label, "new"], ...] 形式を CSV に変換。
    'new' フラグは出力しない。
    """
    rows = []
    for entry in data:
        lat, lon, label = entry[0], entry[1], entry[2]
        rows.append({"latitude": lat, "longitude": lon, "label": label})
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "label"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ---------- 音圧計算 ----------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    各グリッド点で複数スピーカーからの音圧(dB)を合算する簡単な距離減衰モデル。
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, label = spk[0], spk[1], spk[2]
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

# ---------- 等高線(Contour)抽出 ----------
def get_contour_paths(grid_lat, grid_lon, speakers, L0, r_max, levels=None):
    if levels is None:
        levels = sorted([L0-40, L0-30, L0-20, L0-10])
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    try:
        min_val = np.nanmin(sound_grid)
        max_val = np.nanmax(sound_grid)
        if np.isnan(min_val) or np.isnan(max_val) or min_val == max_val:
            return []
    except:
        return []
    fig, ax = plt.subplots()
    try:
        c = ax.contour(grid_lon, grid_lat, sound_grid, levels=levels)
    except Exception:
        plt.close(fig)
        return []
    paths = []
    for level_idx, segs in enumerate(c.allsegs):
        for seg in segs:
            if len(seg) == 0:
                continue
            path = [[p[1], p[0]] for p in seg]  # [lat, lon]
            paths.append({"path": path, "level": levels[level_idx]})
    plt.close(fig)
    return paths

# ---------- 3Dカラム用 ----------
def get_column_data(grid_lat, grid_lon, speakers, L0, r_max):
    """
    3Dカラム表示用に (lat, lon, value, elevation, color) を返す。
    """
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data_list = []
    # 音圧(dB)の min, max を取得
    val_min = np.nanmin(sound_grid)
    val_max = np.nanmax(sound_grid)
    if math.isnan(val_min) or val_min == val_max:
        return pd.DataFrame()  # 空の DataFrame
    
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                # 正規化
                norm = (val - val_min) / (val_max - val_min)
                # 高さ(適宜スケール調整)
                elevation = norm * 1000.0
                # 色付け (R=255*norm, G=255*(1-norm), B=128)
                color = [int(255*norm), int(255*(1-norm)), 128]
                data_list.append({
                    "lat": grid_lat[i, j],
                    "lon": grid_lon[i, j],
                    "value": val,
                    "elevation": elevation,
                    "color": color
                })
    return pd.DataFrame(data_list)

# ---------- Gemini API 関連 ----------
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

def extract_speaker_proposals(response_text):
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

# ---------- メインアプリ (Pydeck) ----------
def main():
    st.title("防災スピーカー音圧可視化マップ (Pydeck) - 3Dカラム＋色分け")

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
        
        if st.session_state.edit_index is not None:
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
        
        display_mode = st.radio("表示モードを選択", ["HeatMap", "Contour Lines", "3D Columns"])
        
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_input("Gemini に問い合わせる内容")
        if st.button("Gemini API を実行"):
            full_prompt = generate_gemini_prompt(gemini_query)
            result = call_gemini_api(full_prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API 実行完了")
            add_speaker_proposals_from_gemini()
    
    # グリッド生成
    lat_min = st.session_state.map_center[0] - 0.01
    lat_max = st.session_state.map_center[0] + 0.01
    lon_min = st.session_state.map_center[1] - 0.01
    lon_max = st.session_state.map_center[1] + 0.01
    Nx, Ny = 60, 60  # グリッド解像度は適宜調整
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, Nx),
        np.linspace(lon_min, lon_max, Ny)
    )
    grid_lat = grid_lat.T
    grid_lon = grid_lon.T
    
    # Pydeck 用に ScatterplotLayer 用の DataFrame 作成
    spk_df = pd.DataFrame(st.session_state.speakers, columns=["lat", "lon", "label", "flag"]) if len(st.session_state.speakers) and len(st.session_state.speakers[0])==4 else None
    if spk_df is None:
        # 長さ4でない場合用の処理
        # すべて [lat, lon, label] 形式として扱う
        spk_list = []
        for s in st.session_state.speakers:
            if len(s) >= 3:
                spk_list.append([s[0], s[1], s[2], ""])
        spk_df = pd.DataFrame(spk_list, columns=["lat", "lon", "label", "flag"])
    else:
        spk_df.fillna({"flag": ""}, inplace=True)
    
    # color を列挙
    def pick_color(row):
        return [255, 0, 0] if row["flag"] == "new" else [0, 0, 255]
    spk_df["color"] = spk_df.apply(pick_color, axis=1)
    
    # ============ ディスプレイモードによる分岐 ============
    if display_mode == "HeatMap":
        # HeatMap用に st.session_state.heatmap_data を DataFrame で保持
        if st.session_state.heatmap_data is None:
            st.session_state.heatmap_data = cached_calculate_heatmap(
                st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
            )
        # HeatMapLayer
        if not st.session_state.heatmap_data.empty:
            # Pydeck HeatmapLayer
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=st.session_state.heatmap_data,
                get_position=["longitude", "latitude"],
                get_weight="weight",
                radiusPixels=50
            )
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=spk_df,
                get_position=["lon", "lat"],
                get_radius=100,
                get_fill_color="color",
                pickable=True
            )
            layers = [heatmap_layer, scatter_layer]
        else:
            st.info("ヒートマップデータが空です。")
            layers = []
    
    elif display_mode == "Contour Lines":
        # Contour
        contour_paths = get_contour_paths(grid_lat, grid_lon, st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
        if contour_paths:
            # PathLayer で等高線を表示
            data_list = []
            for item in contour_paths:
                data_list.append(item)
            contour_layer = pdk.Layer(
                "PathLayer",
                data=data_list,
                get_path="path",
                get_width=2,
                width_min_pixels=2,
                get_color=[255, 0, 0]
            )
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=spk_df,
                get_position=["lon", "lat"],
                get_radius=100,
                get_fill_color="color",
                pickable=True
            )
            layers = [contour_layer, scatter_layer]
        else:
            st.info("有効な等高線が生成されませんでした。")
            layers = []
    
    else:  # "3D Columns"
        # 3Dカラム表示
        col_df = get_column_data(grid_lat, grid_lon, st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
        if col_df.empty:
            st.info("3Dカラム用データが空です。")
            layers = []
        else:
            # ColumnLayer
            column_layer = pdk.Layer(
                "ColumnLayer",
                data=col_df,
                get_position=["lon", "lat"],
                get_elevation="elevation",
                elevation_scale=1,
                radius=30,  # 柱の太さ
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            )
            # スピーカーを点表示
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=spk_df,
                get_position=["lon", "lat"],
                get_radius=100,
                get_fill_color="color",
                pickable=True
            )
            layers = [column_layer, scatter_layer]
    
    # Pydeck のデッキを作成
    if layers:
        view_state = pdk.ViewState(
            latitude=st.session_state.map_center[0],
            longitude=st.session_state.map_center[1],
            zoom=st.session_state.map_zoom,
            pitch=45 if display_mode=="3D Columns" else 0,
            bearing=0
        )
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{label}\n音圧: {value}"}
        )
        st.pydeck_chart(deck)
    else:
        st.write("表示するレイヤーがありません。")
    
    # ダウンロードボタン
    csv_data = export_csv(st.session_state.speakers)
    st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv", "text/csv")
    
    # デバッグ表示
    with st.expander("デバッグ・テスト情報"):
        st.write("スピーカー情報:", st.session_state.speakers)
        if display_mode == "HeatMap" and st.session_state.heatmap_data is not None:
            st.write("ヒートマップデータ件数:", len(st.session_state.heatmap_data))
        st.write("display_mode:", display_mode)
        st.write("L0:", st.session_state.L0)
        st.write("r_max:", st.session_state.r_max)
    
    st.markdown("---")
    st.subheader("Gemini API の回答（テキスト表示）")
    if st.session_state.gemini_result:
        st.text(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答はまだありません。")

# 3Dカラム用データ取得
def get_column_data(grid_lat, grid_lon, speakers, L0, r_max):
    """
    PydeckのColumnLayerで使うためのDataFrameを生成。
    columns: lat, lon, value, elevation, color
    """
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data_list = []
    try:
        val_min = np.nanmin(sound_grid)
        val_max = np.nanmax(sound_grid)
        if math.isnan(val_min) or val_min == val_max:
            return pd.DataFrame()
    except:
        return pd.DataFrame()
    
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                # 0.0～1.0に正規化
                norm = (val - val_min) / (val_max - val_min)
                # 高さ(elevation)は適当にスケール
                elevation = norm * 1000
                # 色: R=255*norm, G=255*(1-norm), B=128
                r = int(255*norm)
                g = int(255*(1 - norm))
                b = 128
                data_list.append({
                    "lat": grid_lat[i, j],
                    "lon": grid_lon[i, j],
                    "value": val,
                    "elevation": elevation,
                    "color": [r, g, b]
                })
    return pd.DataFrame(data_list)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
