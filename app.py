import os
import math
import io
import re
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="上島町全域 + スピーカー方向対応", layout="wide")

# ---------- カスタムCSS ----------
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

# ---------- APIキー設定 ----------
API_KEY = st.secrets["general"]["api_key"]  # secrets.toml の [general] セクション
MODEL_NAME = "gemini-2.0-flash"

# ---------- 方向文字列 → 度数変換 ----------
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
        st.warning(f"方向 '{dir_str}' は不正。0度にします。")
        return 0.0

# ---------- CSV 読み込み/書き出し ----------
def load_csv(file):
    try:
        df = pd.read_csv(file)
        speakers = []
        for idx, row in df.iterrows():
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                label = ""
                if "label" in df.columns and not pd.isna(row["label"]):
                    label = str(row["label"]).strip()
                speakers.append([lat, lon, label])
            except Exception as e:
                st.warning(f"行 {idx+1} 読み込み失敗: {e}")
        return speakers
    except Exception as e:
        st.error(f"CSV読み込みエラー: {e}")
        return []

def export_csv(data):
    rows = []
    for s in data:
        lat, lon, label = s[0], s[1], s[2]
        rows.append({"latitude": lat, "longitude": lon, "label": label})
    df = pd.DataFrame(rows, columns=["latitude","longitude","label"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ---------- 音圧計算(方向対応) ----------
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
    """
    スピーカー情報: [lat, lon, label, direction_deg]
    direction_deg がなければ 0度とみなす。
    コサイン減衰 + 最小0.3倍
    """
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny))
    # Earth distance scale
    for spk in speakers:
        lat_s, lon_s, label = spk[0], spk[1], spk[2]
        # 方向が入っていれば parse, なければ0
        direction_deg = 0.0
        if len(spk) >= 4:
            direction_deg = float(spk[3])  # 4番目を方向度数として扱う
        dlat = grid_lat - lat_s
        dlon = grid_lon - lon_s
        distance = np.sqrt((dlat*111320)**2 + (dlon*111320*math.cos(math.radians(lat_s)))**2)
        distance[distance<1] = 1
        p_db = L0 - 20*np.log10(distance)
        # 方向差計算
        bearing = (np.degrees(np.arctan2(dlon, dlat))) % 360
        angle_diff = np.abs(bearing - direction_deg) % 360
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)
        directional_factor = np.cos(np.radians(angle_diff))
        directional_factor[directional_factor<0.3] = 0.3  # 後方でも0.3倍
        # p_db を 変換 => power
        # directional_factor分パワーを増減
        # p(db) => 10^(p/10)
        # power_sum += directional_factor * 10^(p_db/10)
        p_linear = 10**(p_db/10)
        power = directional_factor * p_linear
        # r_max 超えは 0
        power[distance>r_max] = 0
        power_sum += power
    total_db = np.zeros_like(power_sum)
    mask = (power_sum>0)
    total_db[:] = np.nan
    total_db[mask] = 10*np.log10(power_sum[mask])
    # クリップ
    total_db = np.clip(total_db, L0-40, L0)
    return total_db

def generate_grid_for_kamijima(resolution=80):
    """
    愛媛県上島町全域をカバーする:
      lat ~ 34.20 ~ 34.35
      lon ~ 133.15 ~ 133.28
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
            val = sgrid[i,j]
            data.append({"latitude":grid_lat[i,j],"longitude":grid_lon[i,j],"weight":val})
    return pd.DataFrame(data)

@st.cache_data(show_spinner=False)
def cached_calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon):
    return calculate_heatmap(speakers, L0, r_max, grid_lat, grid_lon)

def get_column_data(grid_lat, grid_lon, speakers, L0, r_max):
    sgrid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    Nx, Ny = grid_lat.shape
    data_list=[]
    val_min=np.nanmin(sgrid)
    val_max=np.nanmax(sgrid)
    if math.isnan(val_min) or val_min==val_max:
        return pd.DataFrame()
    for i in range(Nx):
        for j in range(Ny):
            val=sgrid[i,j]
            norm=(val-val_min)/(val_max-val_min)
            # 高さは低め
            elevation=norm*300
            # 色(半透明)
            r=int(255*norm)
            g=int(255*(1-norm))
            b=128
            a=120
            data_list.append({
                "lat":grid_lat[i,j],
                "lon":grid_lon[i,j],
                "value":val,
                "elevation":elevation,
                "color":[r,g,b,a]
            })
    return pd.DataFrame(data_list)

# ---------- Gemini API関連 ----------
def generate_gemini_prompt(user_query):
    spk_info=""
    if st.session_state.speakers:
        spk_info="\n".join(
            f"{i+1}. 緯度:{s[0]:.6f}, 経度:{s[1]:.6f}, ラベル:{s[2]}"
            for i,s in enumerate(st.session_state.speakers)
        )
    else:
        spk_info="現在、スピーカーは配置されていません。"
    sound_range=f"{st.session_state.L0-40}dB ~ {st.session_state.L0}dB"
    prompt=(
        f"配置スピーカー:\n{spk_info}\n"
        f"音圧範囲: {sound_range}\n"
        "海など設置困難な場所は除外、スピーカー同士は300m以上離れているよう考慮。\n"
        f"ユーザー問い合わせ:{user_query}\n"
        "【座標表記】 緯度:34.255500, 経度:133.207000 で統一。"
    )
    return prompt

def call_gemini_api(query):
    headers={"Content-Type":"application/json"}
    url=f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload={"contents":[{"parts":[{"text":query}]}]}
    try:
        resp=requests.post(url,json=payload,headers=headers,timeout=30)
        resp.raise_for_status()
        rj=resp.json()
        cands=rj.get("candidates",[])
        if not cands:
            st.error("Gemini API: candidates空")
            return "回答なし"
        c0=cands[0]
        cv=c0.get("content","")
        if isinstance(cv,dict):
            parts=cv.get("parts",[])
            txt=" ".join([p.get("text","") for p in parts])
        else:
            txt=str(cv)
        return txt.strip()
    except Exception as e:
        st.error(f"Gemini APIエラー:{e}")
        return f"エラー:{e}"

def extract_speaker_proposals(res_text):
    # 緯度,経度,ラベル,方向など
    pattern=r"(?:緯度[:：]?\s*)([-\d]+\.\d+)[,、\s]+(?:経度[:：]?\s*)([-\d]+\.\d+)(?:[,、\s]+(?:方向[:：]?\s*([^\n]+)))?"
    props=re.findall(pattern,res_text)
    results=[]
    for lat_str,lon_str,dir_str in props:
        try:
            lat=float(lat_str)
            lon=float(lon_str)
            # dir_strはN,E,S... or 数値?
            direction_deg=parse_direction(dir_str) if dir_str else 0.0
            results.append([lat,lon,"Gemini提案",direction_deg])  # 4番目に方向度数
        except:
            continue
    return results

def add_speaker_proposals_from_gemini():
    if not st.session_state.get("gemini_result"):
        st.error("Gemini回答がありません。")
        return
    props=extract_speaker_proposals(st.session_state["gemini_result"])
    if props:
        added=0
        for p in props:
            # p=[lat,lon,label,direction_deg]
            if not any(abs(p[0]-s[0])<1e-6 and abs(p[1]-s[1])<1e-6 for s in st.session_state.speakers):
                st.session_state.speakers.append(p)
                added+=1
        if added>0:
            st.success(f"{added}件のGemini提案スピーカー追加")
            st.session_state.heatmap_data=None
        else:
            st.info("新規スピーカーは見つかりませんでした。")
    else:
        st.info("Gemini回答にスピーカー情報が見つかりません。")

# ---------- メインUI ----------
def main():
    st.title("愛媛県上島町 全域 + スピーカー方向対応")

    if "map_center" not in st.session_state:
        # 上島町中心付近: 34.25,133.20
        st.session_state.map_center=[34.25,133.20]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom=11
    if "speakers" not in st.session_state:
        # 例: [lat,lon,label,direction_deg]
        st.session_state.speakers=[
            [34.25,133.20,"初期スピーカーA",0.0]
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

    with st.sidebar:
        st.header("操作パネル")
        upfile=st.file_uploader("CSVアップロード",type=["csv"])
        if upfile and st.button("CSV登録"):
            new_spk=load_csv(upfile)
            if new_spk:
                st.session_state.speakers.extend(new_spk)
                st.session_state.heatmap_data=None
                st.success(f"{len(new_spk)}件追加")
            else:
                st.error("CSVに有効データなし")
        
        # 手動追加: lat,lon,label,dir
        new_text=st.text_input("スピーカー追加 (lat,lon,label,方向)",placeholder="例:34.25,133.20,役場,N")
        if st.button("追加"):
            parts=new_text.split(",")
            if len(parts)<2:
                st.error("緯度,経度は最低限必要")
            else:
                try:
                    lat=float(parts[0])
                    lon=float(parts[1])
                    label=parts[2].strip() if len(parts)>2 else "新スピーカー"
                    dir_str=parts[3].strip() if len(parts)>3 else "0"
                    dir_deg=parse_direction(dir_str)
                    st.session_state.speakers.append([lat,lon,label,dir_deg])
                    st.session_state.heatmap_data=None
                    st.success(f"追加成功: {lat},{lon},{label},方向:{dir_deg}")
                except Exception as e:
                    st.error(f"追加失敗:{e}")
        
        # 削除・編集
        if st.session_state.speakers:
            opts=[f"{i}:({s[0]:.4f},{s[1]:.4f})[{s[2]}],方向:{s[3]}" for i,s in enumerate(st.session_state.speakers)]
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
                spk=st.session_state.speakers[st.session_state.edit_index]
                new_lat=st.text_input("緯度",value=str(spk[0]))
                new_lon=st.text_input("経度",value=str(spk[1]))
                new_lbl=st.text_input("ラベル",value=spk[2])
                new_dir=st.text_input("方向",value=str(spk[3]))
                if st.form_submit_button("編集保存"):
                    try:
                        latv=float(new_lat)
                        lonv=float(new_lon)
                        labelv=new_lbl
                        dir_deg=parse_direction(new_dir)
                        st.session_state.speakers[st.session_state.edit_index]=[latv,lonv,labelv,dir_deg]
                        st.session_state.heatmap_data=None
                        st.success("編集保存成功")
                        st.session_state.edit_index=None
                    except Exception as e:
                        st.error(f"編集失敗:{e}")

        if st.button("スピーカーリセット"):
            st.session_state.speakers=[]
            st.session_state.heatmap_data=None
            st.success("リセット完了")

        st.session_state.L0=st.slider("初期音圧 (dB)",50,100,st.session_state.L0)
        st.session_state.r_max=st.slider("最大伝播距離 (m)",100,2000,st.session_state.r_max)

        disp_mode=st.radio("表示モード",["HeatMap","3D Columns"])

        st.subheader("Gemini API")
        gem_query=st.text_input("問い合わせ内容")
        if st.button("Gemini API実行"):
            prompt=generate_gemini_prompt(gem_query)
            ans=call_gemini_api(prompt)
            st.session_state.gemini_result=ans
            st.success("Gemini完了")
            add_speaker_proposals_from_gemini()

    # 上島町全域のグリッド生成
    grid_lat, grid_lon = generate_grid_for_kamijima(resolution=80)

    # Pydeck 用スピーカー DF
    # [lat, lon, label, direction_deg]
    spk_list=[]
    for s in st.session_state.speakers:
        lat=s[0]
        lon=s[1]
        lbl=s[2]
        dir_deg=s[3] if len(s)>=4 else 0.0
        spk_list.append([lat,lon,lbl,dir_deg])
    spk_df=pd.DataFrame(spk_list,columns=["lat","lon","label","dir_deg"])
    # 3Dモデルのため z=30
    spk_df["z"]=30
    # newフラグは今回は省略 or labelに"new"とか?
    spk_df["color"]=[ [0,0,255] for _ in range(len(spk_df)) ]

    layers=[]
    if disp_mode=="HeatMap":
        if st.session_state.heatmap_data is None:
            st.session_state.heatmap_data= cached_calculate_heatmap(
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
            st.info("ヒートマップデータが空")
    else:
        # 3D Columns
        col_df=get_column_data(grid_lat,grid_lon,st.session_state.speakers, st.session_state.L0, st.session_state.r_max)
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
                auto_highlight=True
            )
            layers.append(column_layer)
        else:
            st.info("3Dカラムデータ空")

    # スピーカー3Dモデル(上に表示)
    speaker_3d_layer=pdk.Layer(
        "ScenegraphLayer",
        data=spk_df,
        scenegraph="https://raw.githubusercontent.com/visgl/deck.gl-data/master/scenegraph/airplane/scene.gltf",
        get_position=["lon","lat","z"],
        sizeScale=20,
        pickable=True
    )
    layers.append(speaker_3d_layer)

    # 全域音圧範囲
    sgrid=compute_sound_grid(st.session_state.speakers, st.session_state.L0, st.session_state.r_max, grid_lat, grid_lon)
    try:
        dbmin=np.nanmin(sgrid)
        dbmax=np.nanmax(sgrid)
        st.write(f"音圧範囲(上島町全域): {dbmin:.1f} dB ～ {dbmax:.1f} dB")
    except:
        st.write("音圧範囲計算失敗")

    if layers:
        view_state=pdk.ViewState(
            latitude=34.25,  # 上島町中心付近
            longitude=133.20,
            zoom=11,         # やや広域
            pitch=45 if disp_mode=="3D Columns" else 0,
            bearing=0
        )
        deck=pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text":"{label}\n方向:{dir_deg}\n音圧:{value}"}
        )
        st.pydeck_chart(deck)
    else:
        st.write("レイヤーなし")

    csv_data=export_csv(st.session_state.speakers)
    st.download_button("スピーカーCSVダウンロード", csv_data, "speakers.csv","text/csv")

    with st.expander("デバッグ情報"):
        st.write("スピーカー:", st.session_state.speakers)
        st.write("L0:", st.session_state.L0,"r_max:", st.session_state.r_max)
        st.write("表示モード:", disp_mode)

    st.markdown("---")
    st.subheader("Gemini API回答")
    if st.session_state.gemini_result:
        st.text(st.session_state.gemini_result)
    else:
        st.info("Gemini API の回答なし")

if __name__=="__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラー:{e}")
