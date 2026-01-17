import math
import io
import re
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

# ------------------ 設定・定数 ------------------
APP_TITLE    = "上島町 防災無線AI配置シミュレーター (Light Ver.)"
MAP_CENTER   = (34.253, 133.205) # 上島町付近
DEFAULT_ZOOM = 11.5

# デザイン設定
ST_PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# 視認性重視のCSS (白背景・黒文字)
CUSTOM_CSS = """
<style>
    /* 全体のフォントと背景 */
    .stApp { background-color: #FFFFFF; color: #333333; }
    
    /* メトリクス表示の装飾 (明るいグレー背景) */
    div[data-testid="metric-container"] {
        background-color: #F0F2F6;
        border: 1px solid #D1D5DB;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] label {
        color: #555555; /* ラベル色 */
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #000000; /* 数値色 */
    }
    
    /* ボタンのスタイル */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        border: 1px solid #4CAF50;
        color: #4CAF50;
        background-color: white;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #4CAF50;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
    }
    
    /* タイトル周り */
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #2E7D32; }
    h3 { border-left: 5px solid #2E7D32; padding-left: 10px; color: #333; }
    
    /* サイドバーの微調整 */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }
</style>
"""

DIRECTION_MAP = {"N":0, "NE":45, "E":90, "SE":135, "S":180, "SW":225, "W":270, "NW":315}

# ------------------ クラス・ロジック ------------------

class SoundPhysics:
    """音響物理計算エンジン"""
    @staticmethod
    def parse_direction(dir_str: str) -> float:
        s = str(dir_str).strip().upper()
        if s in DIRECTION_MAP: return float(DIRECTION_MAP[s])
        try: return float(s)
        except: return 0.0

    @staticmethod
    def compute_grid(speakers: List[dict], L0: float, r_max: float, beam_width: float, grid_lat: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
        Nx, Ny = grid_lat.shape
        power_sum = np.zeros((Nx, Ny))
        
        m_per_deg_lat = 111000
        m_per_deg_lon = 92000 

        for spk in speakers:
            lat_s, lon_s = spk["lat"], spk["lon"]
            direction = spk["direction"]
            
            dlat = (grid_lat - lat_s) * m_per_deg_lat
            dlon = (grid_lon - lon_s) * m_per_deg_lon
            dist = np.hypot(dlat, dlon)
            dist = np.maximum(dist, 1.0)

            p_db = L0 - 20 * np.log10(dist)
            
            bearing = (np.degrees(np.arctan2(dlon, dlat))) % 360
            angle_diff = np.abs(bearing - direction)
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            
            if beam_width >= 360:
                dir_factor = 1.0
            else:
                norm_angle = np.clip(angle_diff / (beam_width / 2), 0, 2)
                dir_factor = np.cos(norm_angle * (np.pi / 2)) 
                dir_factor = np.clip(dir_factor, 0.1, 1.0)

            power = (10**(p_db/10)) * dir_factor
            power[dist > r_max] = 0
            power_sum += power

        total_db = 10 * np.log10(power_sum + 1e-12)
        return np.where(power_sum > 0, np.clip(total_db, 0, L0), np.nan)

class IntelligentPlanner:
    """人智を超えるための分析ロジック"""
    
    @staticmethod
    def find_blind_spot(grid_val: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray, threshold_db: float) -> dict:
        """死角検知（NumPy版）"""
        val_filled = np.nan_to_num(grid_val, nan=0.0)
        silent_mask = (val_filled > 0) & (val_filled < threshold_db)
        y_idxs, x_idxs = np.where(silent_mask)
        
        if len(y_idxs) == 0:
            y_idxs, x_idxs = np.where(val_filled == 0)
            
        if len(y_idxs) == 0:
            return None

        cy = np.mean(y_idxs)
        cx = np.mean(x_idxs)
        lat_idx, lon_idx = int(cy), int(cx)
        lat_idx = min(lat_idx, grid_lat.shape[0]-1)
        lon_idx = min(lon_idx, grid_lat.shape[1]-1)

        return {
            "lat": grid_lat[lat_idx, lon_idx],
            "lon": grid_lon[lat_idx, lon_idx],
            "score": len(y_idxs)
        }

    @staticmethod
    def generate_gemini_prompt(query: str, speakers: List[dict], blind_spot: dict, L0: float) -> str:
        spk_list = "\n".join([f"- {s['label']}: ({s['lat']:.5f}, {s['lon']:.5f}) {s['direction']}°" for s in speakers])
        
        blind_info = ""
        if blind_spot:
            blind_info = (
                f"\n【システム分析による重要死角】\n"
                f"緯度: {blind_spot['lat']:.6f}, 経度: {blind_spot['lon']:.6f} 付近\n"
            )

        return (
            "あなたは日本の地方自治体（上島町）の防災無線計画を支援する「高度防災コンサルタントAI」です。\n"
            "学習済みの地理情報と、以下のシミュレーション結果を統合して回答してください。\n\n"
            "## 現状の配置\n"
            f"{spk_list}\n"
            f"出力音圧: {L0}dB\n"
            f"{blind_info}\n"
            "## ユーザーの指示\n"
            f"{query}\n\n"
            "## ミッション\n"
            "1. 死角の位置が地理的に設置可能か（海上や断崖でないか）判定。\n"
            "2. 設置不可能な場合、現実的な代替地点を提案。\n"
            "3. 新設する場合の最適な「緯度」「経度」「方向(0-360)」「推奨理由」を回答。\n"
            "4. 回答の最後に必ずJSON形式で提案座標を出力。\n"
            "例: ```json\n{\"lat\": 34.123, \"lon\": 133.456, \"direction\": 180, \"label\": \"AI提案地点\"}\n```"
        )

# ------------------ UIコンポーネント ------------------

def render_sidebar():
    st.sidebar.title("🛠 設定パネル")
    
    uploaded_file = st.sidebar.file_uploader("CSVインポート", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if {'latitude', 'longitude'}.issubset(df.columns):
            new_spks = []
            for _, row in df.iterrows():
                new_spks.append({
                    "lat": row["latitude"], "lon": row["longitude"],
                    "label": row.get("label", "No Name"),
                    "direction": SoundPhysics.parse_direction(row.get("direction", 0))
                })
            st.session_state.speakers = new_spks
            st.toast(f"{len(new_spks)}件読み込み完了", icon="📂")

    st.sidebar.divider()
    
    with st.sidebar.expander("📡 音響パラメータ", expanded=False):
        L0 = st.slider("出力音圧 (dB)", 70, 130, 85)
        r_max = st.slider("最大到達距離 (m)", 100, 3000, 800)
        beam = st.slider("指向性ビーム幅 (度)", 30, 360, 120)
    
    st.session_state.params = {"L0": L0, "r_max": r_max, "beam": beam}

    st.sidebar.divider()
    
    with st.sidebar.form("add_speaker"):
        st.write("手動追加")
        c1, c2 = st.columns(2)
        lat = c1.number_input("緯度", value=MAP_CENTER[0], format="%.6f")
        lon = c2.number_input("経度", value=MAP_CENTER[1], format="%.6f")
        label_txt = st.text_input("名称", "新規スピーカー")
        direct = st.number_input("方向", 0, 360, 0)
        if st.form_submit_button("追加"):
            st.session_state.speakers.append({
                "lat": lat, "lon": lon, "label": label_txt, "direction": direct
            })
            st.rerun()

    if st.sidebar.button("全データクリア", type="primary"):
        st.session_state.speakers = []
        st.session_state.proposals = []
        st.rerun()

def call_gemini_api(prompt):
    api_key = st.secrets["general"].get("api_key")
    if not api_key:
        st.error("SecretsにAPIキーがありません")
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        res = requests.post(url, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        st.error(f"AI通信エラー: {e}")
        return None

# ------------------ メイン処理 ------------------

def main():
    st.set_page_config(**ST_PAGE_CONFIG)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "speakers" not in st.session_state:
        st.session_state.speakers = [
            {"lat": 34.253, "lon": 133.205, "label": "役場本庁舎", "direction": 0},
            {"lat": 34.248, "lon": 133.200, "label": "港湾施設", "direction": 180}
        ]
    if "proposals" not in st.session_state:
        st.session_state.proposals = []

    render_sidebar()

    st.title("🔊 上島町 防災無線配置シミュレーター")
    
    params = st.session_state.params
    resolution = 100
    
    if st.session_state.speakers:
        lats = [s['lat'] for s in st.session_state.speakers]
        lons = [s['lon'] for s in st.session_state.speakers]
        lat_min, lat_max = min(lats)-0.02, max(lats)+0.02
        lon_min, lon_max = min(lons)-0.02, max(lons)+0.02
    else:
        lat_min, lat_max = MAP_CENTER[0]-0.02, MAP_CENTER[0]+0.02
        lon_min, lon_max = MAP_CENTER[1]-0.02, MAP_CENTER[1]+0.02

    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, resolution),
        np.linspace(lon_min, lon_max, resolution),
        indexing="ij"
    )
    
    sound_grid = SoundPhysics.compute_grid(
        st.session_state.speakers, 
        params["L0"], params["r_max"], params["beam"], 
        grid_lat, grid_lon
    )
    
    blind_spot = IntelligentPlanner.find_blind_spot(sound_grid, grid_lat, grid_lon, threshold_db=60)

    m1, m2, m3, m4 = st.columns(4)
    valid_cells = np.count_nonzero(~np.isnan(sound_grid))
    covered_cells = np.count_nonzero(np.nan_to_num(sound_grid, 0) >= 60)
    coverage_rate = (covered_cells / valid_cells * 100) if valid_cells > 0 else 0
    
    m1.metric("設置数", f"{len(st.session_state.speakers)} 基")
    m2.metric("有効カバー率", f"{coverage_rate:.1f} %")
    m3.metric("最大到達距離", f"{params['r_max']} m")
    m4.metric("死角検知", "あり" if blind_spot else "なし", delta_color="inverse" if blind_spot else "normal")

    tab_map, tab_ai = st.tabs(["🗺️ シミュレーションマップ", "🤖 AI配置コンサルタント"])

    with tab_map:
        heatmap_data = []
        mask = ~np.isnan(sound_grid)
        for i, j in np.argwhere(mask):
            val = sound_grid[i, j]
            heatmap_data.append([grid_lon[i, j], grid_lat[i, j], val])
        
        df_heat = pd.DataFrame(heatmap_data, columns=["lon", "lat", "weight"])

        layers = []
        # ヒートマップ層 (色調整済み)
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=df_heat,
            get_position=["lon", "lat"],
            get_weight="weight",
            radius_pixels=40,
            intensity=1,
            threshold=0.3,
            opacity=0.5, # 白背景でも見えるように調整
            color_range=[
                [0, 0, 255, 100],     # 青
                [0, 255, 0, 150],     # 緑
                [255, 255, 0, 180],   # 黄
                [255, 0, 0, 200]      # 赤
            ]
        ))
        
        df_spk = pd.DataFrame(st.session_state.speakers)
        if not df_spk.empty:
            # スピーカーマーカー (白背景で見える濃い青に)
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_spk,
                get_position=["lon", "lat"],
                get_fill_color=[0, 80, 200], # 濃い青
                get_radius=50,
                pickable=True,
            ))
            # ラベル (黒文字)
            layers.append(pdk.Layer(
                "TextLayer",
                data=df_spk,
                get_position=["lon", "lat"],
                get_text="label",
                get_size=15,
                get_color=[0, 0, 0], # 黒
                get_weight=700,
                get_alignment_baseline="'bottom'",
                get_pixel_offset=[0, -10]
            ))

        if blind_spot:
            df_blind = pd.DataFrame([blind_spot])
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_blind,
                get_position=["lon", "lat"],
                get_fill_color=[200, 50, 200],
                get_line_color=[0, 0, 0], # 枠線を黒に
                get_line_width=2,
                get_radius=100,
                stroked=True,
                pickable=True,
            ))
            
        if st.session_state.proposals:
             df_prop = pd.DataFrame(st.session_state.proposals)
             layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_prop,
                get_position=["lon", "lat"],
                get_fill_color=[0, 200, 100],
                get_radius=80,
                pickable=True,
                stroked=True,
                get_line_color=[0,0,0],
                get_line_width=3
            ))

        view_state = pdk.ViewState(
            latitude=np.mean(lats) if st.session_state.speakers else MAP_CENTER[0],
            longitude=np.mean(lons) if st.session_state.speakers else MAP_CENTER[1],
            zoom=DEFAULT_ZOOM,
            pitch=0
        )
        
        # マップスタイルをライトモード(light-v9)に変更
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=view_state,
            layers=layers,
            tooltip={"text": "{label}\n音圧: {weight}dB"}
        ))
        
        st.caption("紫: 死角重心 | 緑: AI提案 | マップスタイル: Light Mode")

    with tab_ai:
        c_ai_1, c_ai_2 = st.columns([1, 2])
        
        with c_ai_1:
            st.subheader("AI コンサルタント")
            st.info("AIが地形と音響シミュレーションを解析し、最適な場所を提案します。")
            user_query = st.text_area("指示・条件", "死角を解消するための最適な場所を1つ提案して。", height=100)
            
            if st.button("🚀 配置案を作成"):
                with st.spinner("AI解析中..."):
                    prompt = IntelligentPlanner.generate_gemini_prompt(
                        user_query, st.session_state.speakers, blind_spot, params["L0"]
                    )
                    response_text = call_gemini_api(prompt)
                    
                    if response_text:
                        st.session_state.last_response = response_text
                        json_match = re.search(r"```json\s*({.*?})\s*```", response_text, re.DOTALL)
                        if json_match:
                            try:
                                import json
                                prop_data = json.loads(json_match.group(1))
                                st.session_state.proposals = [prop_data]
                                st.success("提案を作成しました")
                            except:
                                st.warning("座標抽出失敗")
        
        with c_ai_2:
            st.subheader("AI 分析レポート")
            if "last_response" in st.session_state:
                st.markdown(st.session_state.last_response)
                if st.session_state.proposals:
                    p = st.session_state.proposals[0]
                    if st.button("この提案を採用する"):
                        st.session_state.speakers.append(p)
                        st.session_state.proposals = []
                        st.session_state.last_response = ""
                        st.rerun()
            else:
                st.write("ここに分析結果が表示されます。")

if __name__ == "__main__":
    main()
