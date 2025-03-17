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

st.set_page_config(page_title="防災スピーカー音圧可視化 (Pydeck 3D)", layout="wide")

# カスタムCSS
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

API_KEY = st.secrets["general"]["api_key"]
MODEL_NAME = "gemini-2.0-flash"

# ---------------- CSV読み込み・書き出し ---------------
def load_csv(file):
    """CSVから[[lat,lon,label],...]形式を読み込む"""
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

def export_csv(data):
    """スピーカー情報をCSVに変換"""
    rows = []
    for s in data:
        lat, lon, label = s[0], s[1], s[2]
        rows.append({"latitude": lat, "longitude": lon, "label": label})
    df = pd.DataFrame(rows, columns=["latitude","longitude","label"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ---------------- 音圧計算 ---------------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    距離減衰モデル (L0-20log10(r))。
    音圧を L0-40～L0 にクリップして、遠方も最低値のカラムを表示。
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    for spk in speakers:
        lat, lon, _ = spk
        dlat = grid_lat - lat
        dlon = grid_lon - lon
        distance = np.sqrt((dlat * 111320)**2 + (dlon * 111320 * math.cos(math.radians(lat)))**2)
        distance[distance<1] = 1
        p_db = L0 - 20*np.log10(distance)
        # クリップ
        p_db = np.clip(p_db, L0-40, L0)
        power_sum += 10**(p_db/10)
    total_db = 10*np.log10(power_sum)
    total_db = np.clip(total_db, L0-40, L0)
    return total_db

def generate_grid(center, delta=0.01, resolution=60):
    lat, lon = center
    lat_min, lat_max = lat-delta, lat+delta
    lon_min, lon_max = lon-delta, lon+delta
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, resolution),
        np.linspace(lon_min, lon_max, resolution)
    )
    return grid_lat.T, grid_lon.T

def calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            data.append({"latitude": grid_lat[i,j], "longitude": grid_lon[i,j], "weight": val})
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

def get_column_data(grid_lat, grid_lon, speakers, L0, r_max):
    """
    ColumnLayer用: 各グリッド点 (lat, lon) に音圧(dB)を割り当て、
    弱→青, 強→赤 (半透明) かつ高さを抑える
    """
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data_list = []
    val_min = np.nanmin(sound_grid)
    val_max = np.nanmax(sound_grid)
    if math.isnan(val_min) or val_min==val_max:
        return pd.DataFrame()
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i,j]
            norm = (val - val_min)/(val_max - val_min)
            # 高さ (抑えめ)
            elevation = norm*300.0
            # 色: RGBA, 0→青,1→赤, α=120
            r = int(255*norm)
            g = int(255*(1 - norm))
            b = 128
            a = 120
            data_list.append({
                "lat": grid_lat[i,j],
                "lon": grid_lon[i,j],
                "value": val,
                "elevation": elevation,
                "color": [r,g,b,a]
            })
    return pd.DataFrame(data_list)

# ---------------- Gemini API ---------------
def generate_gemini_prompt(user_query):
    spk_info = ""
    if st.session_state.speakers:
        spk_info = "\n".join(
            f"{i+1}. 緯度: {s[0]:.6f}, 経度: {s[1]:.6f}, ラベル: {s[2]}"
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
        "【座標表記形式】 緯度: 34.255500, 経度: 133.207000 で統一してください。"
    )
    return prompt

def call_gemini_api(query):
    headers = {"Content-Type":"application/json"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload = {"contents":[{"parts":[{"text":query}]}]}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        rjson = r.json()
        candidates = rjson.get("candidates", [])
        if not candidates:
            st.error("Gemini API エラー: candidatesが空")
            return "回答が得られませんでした。"
        c0 = candidates[0]
        content_val = c0.get("content","")
        if isinstance(content_val, dict):
            parts = content_val.get("parts",[])
            text = " ".join([p.get("text","") for p in parts])
        else:
            text = str(content_val)
        return text.strip()
    except Exception as e:
        st.error(f"Gemini API呼び出しエラー: {e}")
        return f"エラー: {e}"

def extract_speaker_proposals(response_text):
    pattern = r"(?:緯度[:：]?\s*)([-\d]+\.\d+)[,、\s]+(?:経度[:：]?\s*)([-\d]+\.\d+)(?:[,、\s]+(?:方向[:：]?\s*[-\d\.]+))?(?:[,、\s]+(?:ラベル[:：]?\s*([^\n]+))?)?"
    props = re.findall(pattern, response_text)
    results = []
    for lat_str, lon_str, label in props:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            lbl = label.strip() if label else ""
            results.append([lat, lon, lbl, "new"])
        except:
            continue
    return results

def add_speaker_proposals_from_gemini():
    if not st.session_state.get("gemini_result"):
        st.error("Gemini API の回答がありません。")
        return
    proposals = extract_speaker_proposals(st.session_state["gemini_result"])
    if proposals:
        added_count = 0
        for p in proposals:
            if not any(abs(p[0] - s[0])<1e-6 and abs(p[1] - s[1])<1e-6 for s in st.session_state.speakers):
                st.session_state.speakers.append(p)
                added_count += 1
        if added_count>0:
            st.success(f"Gemini の回答から {added_count} 件の新規スピーカー情報を追加しました。")
            st.session_state.heatmap_data = None
        else:
            st.info("Gemini の回答から新たなスピーカー情報は見つかりませんでした。")
    else:
        st.info("Gemini の回答からスピーカー情報の抽出に失敗しました。")

# ---------------- ScenegraphLayer (3Dスピーカー) ---------------
def create_speaker_3d_layer(spk_df):
    """
    ScenegraphLayerを使って、スピーカー位置に3Dモデルを表示。
    ここではデモ用に公開URLの飛行機モデルを使用。
    'z'カラムを使って高さを少し上げるなど可能。
    """
    # 3DモデルURL (Deck.gl公式サンプル: airplane)
    SCENEGRAPH_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/scenegraph/airplane/scene.gltf"
    # スピーカーを少し上に配置したければ z=30 など追加
    spk_df["z"] = 30  # カラム追加: 3Dモデルを上に浮かせる
    return pdk.Layer(
        "ScenegraphLayer",
        data=spk_df,
        scenegraph=SCENEGRAPH_URL,
        get_position=["lon","lat","z"],  # 3次元座標
        get_orientation=[0,0,0],
        sizeScale=20,  # 大きさ
        pickable=True
    )

# ---------------- メインアプリ ---------------
def main():
    st.title("防災スピーカー音圧可視化 (Pydeck) - 3Dスピーカー＋3Dカラム")

    if "map_center" not in st.session_state:
        st.session_state.map_center = [34.25741795269067,133.20450105700033]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom=14
    if "speakers" not in st.session_state:
        st.session_state.speakers = [
            [34.25741795269067,133.20450105700033,"初期スピーカーA"],
            [34.2574617056359,133.204487449849,"初期スピーカーB"]
        ]
    if "heatmap_data" not in st.session_state:
        st.session_state.heatmap_data=None
    if "L0" not in st.session_state:
        st.session_state.L0=80
    if "r_max" not in st.session_state:
        st.session_state.r_max=500
    if "gemini_result" not in st.session_state:
        st.session_state.gemini_result=None
    if "edit_index" not in st.session_state:
        st.session_state.edit_index=None

    # サイドバー
    with st.sidebar:
        st.header("操作パネル")
        # CSV
        upfile = st.file_uploader("CSVファイルをアップロード",type=["csv"])
        if upfile and st.button("CSVからスピーカー登録"):
            new_spk = load_csv(upfile)
            if new_spk:
                st.session_state.speakers.extend(new_spk)
                st.session_state.heatmap_data=None
                st.success(f"CSVから {len(new_spk)} 件のスピーカーを追加しました。")
            else:
                st.error("正しいCSVが見つかりませんでした。")
        # 手動追加
        spk_input = st.text_input("スピーカー追加 (lat,lon,label)",placeholder="例:34.2579,133.2072,役場")
        if st.button("スピーカー追加"):
            parts=spk_input.split(",")
            if len(parts)<2:
                st.error("形式が不正です。(lat,lon,label)")
            else:
                try:
                    lat=float(parts[0])
                    lon=float(parts[1])
                    label=parts[2].strip() if len(parts)>2 else ""
                    st.session_state.speakers.append([lat,lon,label])
                    st.session_state.heatmap_data=None
                    st.success(f"追加成功: 緯度{lat},経度{lon},ラベル:{label}")
                except Exception as e:
                    st.error(f"追加エラー:{e}")
        # 削除・編集
        if st.session_state.speakers:
            opts=[f"{i}:({s[0]:.6f},{s[1]:.6f})-{s[2]}" for i,s in enumerate(st.session_state.speakers)]
            sel=st.selectbox("スピーカー選択",list(range(len(opts))),format_func=lambda i:opts[i])
            c1,c2=st.columns(2)
            with c1:
                if st.button("選択削除"):
                    try:
                        del st.session_state.speakers[sel]
                        st.session_state.heatmap_data=None
                        st.success("削除成功")
                    except Exception as e:
                        st.error(f"削除失敗:{e}")
            with c2:
                if st.button("選択編集"):
                    st.session_state.edit_index=sel
        else:
            st.info("スピーカーがありません。")

        if st.session_state.edit_index is not None:
            with st.form("edit_form"):
                spk = st.session_state.speakers[st.session_state.edit_index]
                new_lat=st.text_input("緯度",value=str(spk[0]))
                new_lon=st.text_input("経度",value=str(spk[1]))
                new_lbl=st.text_input("ラベル",value=spk[2])
                if st.form_submit_button("編集保存"):
                    try:
                        latv=float(new_lat)
                        lonv=float(new_lon)
                        st.session_state.speakers[st.session_state.edit_index]=[latv,lonv,new_lbl]
                        st.session_state.heatmap_data=None
                        st.success("編集保存成功")
                        st.session_state.edit_index=None
                    except Exception as e:
                        st.error(f"編集保存エラー:{e}")
        
        if st.button("スピーカーリセット"):
            st.session_state.speakers=[]
            st.session_state.heatmap_data=None
            st.success("リセット完了")

        # パラメータ
        st.session_state.L0 = st.slider("初期音圧レベル(dB)",50,100,st.session_state.L0)
        st.session_state.r_max = st.slider("最大伝播距離(m)",100,2000,st.session_state.r_max)

        # 表示モード
        disp_mode=st.radio("表示モード",["HeatMap","3D Columns"])

        # Gemini
        st.subheader("Gemini API")
        gem_query=st.text_input("問い合わせ内容")
        if st.button("Gemini API 実行"):
            prompt=generate_gemini_prompt(gem_query)
            ans=call_gemini_api(prompt)
            st.session_state.gemini_result=ans
            st.success("Gemini API完了")
            add_speaker_proposals_from_gemini()

    # グリッド生成
    grid_lat, grid_lon = generate_grid(st.session_state.map_center, delta=0.01, resolution=60)
    
    # スピーカー DataFrame
    spk_list=[]
    for s in st.session_state.speakers:
        flag=s[3] if len(s)>=4 else ""
        spk_list.append([s[0],s[1],s[2],flag])
    spk_df=pd.DataFrame(spk_list,columns=["lat","lon","label","flag"])
    # ScenegraphLayer用にz座標
    spk_df["z"] = 30  # カラム追加: 3Dモデルを上に表示
    # 色は new->赤, else->青
    def pick_color(f):
        return [255,0,0] if f=="new" else [0,0,255]
    spk_df["color"]=spk_df["flag"].apply(pick_color)

    # レイヤー
    layers=[]

    if disp_mode=="HeatMap":
        # ヒートマップ
        if st.session_state.heatmap_data is None:
            st.session_state.heatmap_data = cached_calculate_heatmap(
                st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon
            )
        if not st.session_state.heatmap_data.empty:
            heatmap_layer=pdk.Layer(
                "HeatmapLayer",
                data=st.session_state.heatmap_data,
                get_position=["longitude","latitude"],
                get_weight="weight",
                radiusPixels=50,
                min_opacity=0.1,
                max_opacity=0.3
            )
            layers.append(heatmap_layer)
        else:
            st.info("ヒートマップデータが空です。")
    else:
        # 3Dカラム
        col_df = get_column_data(grid_lat,grid_lon, st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
        if not col_df.empty:
            column_layer=pdk.Layer(
                "ColumnLayer",
                data=col_df,
                get_position=["lon","lat"],
                get_elevation="elevation",
                elevation_scale=1,
                radius=20,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            )
            layers.append(column_layer)
        else:
            st.info("3Dカラム用データが空です。")

    # スピーカー 3Dモデル (ScenegraphLayer)
    # カラムより後に追加 => 上に表示
    scene_layer=pdk.Layer(
        "ScenegraphLayer",
        data=spk_df,
        scenegraph="https://raw.githubusercontent.com/visgl/deck.gl-data/master/scenegraph/airplane/scene.gltf",
        get_position=["lon","lat","z"],
        get_orientation=[0,0,0],
        sizeScale=20,  # 大きさ
        pickable=True,
    )
    layers.append(scene_layer)

    # 全域音圧範囲
    sgrid=compute_sound_grid(st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon)
    try:
        dbmin=np.nanmin(sgrid)
        dbmax=np.nanmax(sgrid)
        st.write(f"全スピーカーの音圧範囲: {dbmin:.1f} dB ～ {dbmax:.1f} dB")
    except:
        st.write("音圧範囲の計算失敗")

    # Pydeck表示
    if layers:
        view_state=pdk.ViewState(
            latitude=st.session_state.map_center[0],
            longitude=st.session_state.map_center[1],
            zoom=st.session_state.map_zoom,
            pitch=45 if disp_mode=="3D Columns" else 0,
            bearing=0
        )
        deck=pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text":"{label}\n音圧:{value}"}
        )
        st.pydeck_chart(deck)
    else:
        st.write("表示するレイヤーがありません。")

    # CSVダウンロード
    csv_data=export_csv(st.session_state.speakers)
    st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv","text/csv")

    with st.expander("デバッグ情報"):
        st.write("スピーカー:", st.session_state.speakers)
        st.write("表示モード:", disp_mode)
        st.write("L0:", st.session_state.L0, "r_max:", st.session_state.r_max)
    
    st.markdown("---")
    st.subheader("Gemini API の回答（テキスト表示）")
    if st.session_state.gemini_result:
        st.text(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答はまだありません。")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
