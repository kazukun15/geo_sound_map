import math
import io
import re
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk

# Foliumのインポート試行（マップクリック機能用）
try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ------------------ 設定・定数 ------------------
APP_TITLE    = "上島町 防災無線AI配置シミュレーター (God Mode v2)"
MAP_CENTER   = (34.253, 133.205) # 上島町（弓削島）付近
DEFAULT_ZOOM = 11.5

# デザイン設定
ST_PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    /* Metric Card */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #41424C;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,255,100,0.2);
    }
    /* Headers */
    h1 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #00FF94; }
    h3 { border-left: 5px solid #00FF94; padding-left: 10px; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E1E1E;
        border-radius: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00FF94 !important;
        color: #000000 !important;
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
        silent_mask = (val_filled < threshold_db)
        
        y_idxs, x_idxs = np.where(silent_mask)
        
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
        spk_list_str = "\n".join([f"- {s['label']}: ({s['lat']:.5f}, {s['lon']:.5f}) {s['direction']}°" for s in speakers])
        
        blind_info = ""
        if blind_spot:
            blind_info = (
                f"\n【システム分析による重要死角重心】\n"
                f"緯度: {blind_spot['lat']:.6f}, 経度: {blind_spot['lon']:.6f} 付近\n"
                f"このエリアは現在、十分な音圧が確保されていない空白地帯の中心です。\n"
            )

        return (
            "あなたは日本の地方自治体（上島町）の防災無線計画を支援する「高度防災コンサルタントAI」です。\n"
            "地形、集落の分布、避難経路などの地理的知識（あなたの学習データ）と、以下のシミュレーション結果を統合して回答してください。\n\n"
            "## 現状の配置\n"
            f"{spk_list_str}\n"
            f"出力音圧: {L0}dB\n"
            f"{blind_info}\n"
            "## ユーザーの指示\n"
            f"{query}\n\n"
            "## ミッション\n"
            "1. 上記の「重要死角」の位置が、地理的に設置可能か（海上や断崖絶壁でないか）判定してください。\n"
            "2. もし設置不可能な場合、近くの道路沿いや施設など、現実的な代替地点を提案してください。\n"
            "3. 新設する場合の最適な「緯度」「経度」「方向(0-360)」「推奨理由」を答えてください。\n"
            "4. 回答の最後に必ずJSON形式で提案座標を出力してください。\n"
            "例: ```json\n{\"lat\": 34.123, \"lon\": 133.456, \"direction\": 180, \"label\": \"AI提案地点\"}\n```"
        )

# ------------------ UIコンポーネント ------------------

def render_sidebar():
    st.sidebar.title("🛠 設定パネル")
    
    # CSVインポート
    uploaded_file = st.sidebar.file_uploader("CSVインポート", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required = {'latitude', 'longitude'}
            if required.issubset(df.columns):
                new_spks = []
                for _, row in df.iterrows():
                    new_spks.append({
                        "lat": row["latitude"], "lon": row["longitude"],
                        "label": row.get("label", "No Name"),
                        "direction": SoundPhysics.parse_direction(row.get("direction", 0))
                    })
                st.session_state.speakers = new_spks
                st.toast(f"{len(new_spks)}件のデータを読み込みました", icon="📂")
        except Exception as e:
            st.sidebar.error(f"ファイル読み込みエラー: {e}")

    st.sidebar.divider()
    
    # パラメータ設定
    with st.sidebar.expander("📡 音響パラメータ調整", expanded=False):
        L0 = st.slider("出力音圧 (dB)", 70, 130, 85)
        r_max = st.slider("最大到達距離 (m)", 100, 3000, 800)
        beam = st.slider("指向性ビーム幅 (度)", 30, 360, 120)
    
    st.session_state.params = {"L0": L0, "r_max": r_max, "beam": beam}

    st.sidebar.divider()
    
    # リセットボタン
    if st.sidebar.button("全データクリア", type="primary"):
        st.session_state.speakers = []
        st.session_state.proposals = []
        st.rerun()

def call_gemini_api_robust(prompt):
    """リトライ機能付きAPI呼び出し (タイムアウト対策版)"""
    api_key = st.secrets["general"].get("api_key")
    if not api_key:
        st.error("SecretsにAPIキーが設定されていません。")
        return None
    
    model_name = "gemini-2.0-flash-exp" 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    # タイムアウトを120秒に設定
    TIMEOUT_SEC = 120 
    max_retries = 3 
    
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, timeout=TIMEOUT_SEC)
            
            # レート制限 (429)
            if res.status_code == 429:
                wait_time = 2 ** (attempt + 1)
                st.toast(f"AIアクセス集中: {wait_time}秒待機して再試行します...", icon="⏳")
                time.sleep(wait_time)
                continue
            
            res.raise_for_status()
            
            data = res.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                return None
            
        except requests.exceptions.Timeout:
            st.error("AIからの応答がタイムアウトしました (120秒)。")
            return None
        except Exception as e:
            # 最後の試行でなければログだけ出してリトライ
            if attempt == max_retries - 1:
                st.error(f"AIエラー: {e}")
                return None
            time.sleep(2)
            
    return None

# ------------------ メイン処理 ------------------

def main():
    st.set_page_config(**ST_PAGE_CONFIG)
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # セッション初期化
    if "speakers" not in st.session_state:
        st.session_state.speakers = [
            {"lat": 34.253, "lon": 133.205, "label": "役場本庁舎", "direction": 0},
            {"lat": 34.248, "lon": 133.200, "label": "港湾施設", "direction": 180}
        ]
    if "proposals" not in st.session_state:
        st.session_state.proposals = []
    # マップクリック座標用
    if "clicked_coords" not in st.session_state:
        st.session_state.clicked_coords = None

    render_sidebar()

    st.title("🔊 上島町 防災無線配置シミュレーター")
    
    params = st.session_state.params
    resolution = 100
    
    # 範囲決定
    if st.session_state.speakers:
        lats = [s['lat'] for s in st.session_state.speakers]
        lons = [s['lon'] for s in st.session_state.speakers]
        if not lats:
             center_lat, center_lon = MAP_CENTER
             lat_min, lat_max = center_lat-0.02, center_lat+0.02
             lon_min, lon_max = center_lon-0.02, center_lon+0.02
        else:
            center_lat, center_lon = np.mean(lats), np.mean(lons)
            lat_min, lat_max = min(lats)-0.02, max(lats)+0.02
            lon_min, lon_max = min(lons)-0.02, max(lons)+0.02
    else:
        center_lat, center_lon = MAP_CENTER
        lat_min, lat_max = MAP_CENTER[0]-0.02, MAP_CENTER[0]+0.02
        lon_min, lon_max = MAP_CENTER[1]-0.02, MAP_CENTER[1]+0.02

    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, resolution),
        np.linspace(lon_min, lon_max, resolution),
        indexing="ij"
    )
    
    # 計算実行
    sound_grid = SoundPhysics.compute_grid(
        st.session_state.speakers, 
        params["L0"], params["r_max"], params["beam"], 
        grid_lat, grid_lon
    )
    
    blind_spot = IntelligentPlanner.find_blind_spot(sound_grid, grid_lat, grid_lon, threshold_db=60)

    # メトリクス
    m1, m2, m3, m4 = st.columns(4)
    valid_cells = np.count_nonzero(~np.isnan(sound_grid))
    covered_cells = np.count_nonzero(np.nan_to_num(sound_grid, 0) >= 60)
    coverage_rate = (covered_cells / valid_cells * 100) if valid_cells > 0 else 0
    
    m1.metric("設置数", f"{len(st.session_state.speakers)} 基")
    m2.metric("有効カバー率 (60dB以上)", f"{coverage_rate:.1f} %")
    m3.metric("最大到達距離", f"{params['r_max']} m")
    m4.metric("重要死角検知", "あり" if blind_spot else "なし", delta_color="inverse" if blind_spot else "normal")

    # --- タブ構成 ---
    tab_map, tab_add, tab_ai = st.tabs(["🗺️ シミュレーション", "📍 スピーカー追加", "🤖 AIコンサルタント"])

    # 1. シミュレーションマップ (PyDeck)
    with tab_map:
        heatmap_data = []
        mask = ~np.isnan(sound_grid)
        for i, j in np.argwhere(mask):
            val = sound_grid[i, j]
            heatmap_data.append([grid_lon[i, j], grid_lat[i, j], val])
        
        df_heat = pd.DataFrame(heatmap_data, columns=["lon", "lat", "weight"])

        layers = []
        layers.append(pdk.Layer(
            "HeatmapLayer",
            data=df_heat,
            get_position=["lon", "lat"],
            get_weight="weight",
            radius_pixels=40,
            intensity=1,
            threshold=0.3,
            opacity=0.6,
            color_range=[[0, 255, 255, 50], [0, 255, 0, 100], [255, 255, 0, 150], [255, 0, 0, 200]]
        ))
        
        df_spk = pd.DataFrame(st.session_state.speakers)
        if not df_spk.empty:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=df_spk,
                get_position=["lon", "lat"],
                get_fill_color=[255, 255, 255],
                get_radius=50,
                pickable=True,
            ))
            layers.append(pdk.Layer(
                "TextLayer",
                data=df_spk,
                get_position=["lon", "lat"],
                get_text="label",
                get_size=14,
                get_color=[255, 255, 255],
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
                get_line_color=[255, 255, 255],
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
                get_fill_color=[0, 255, 127],
                get_radius=80,
                pickable=True,
                stroked=True,
                get_line_color=[255,255,255],
                get_line_width=3
            ))

        view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=DEFAULT_ZOOM, pitch=0)
        
        st.pydeck_chart(pdk.Deck(
            map_style=pdk.map_styles.CARTO_DARK,
            initial_view_state=view_state,
            layers=layers,
            tooltip={"text": "{label}\n音圧: {weight}dB"}
        ))

    # 2. 直感的な追加マップ (Folium)
    with tab_add:
        c_add_L, c_add_R = st.columns([2, 1])
        
        with c_add_L:
            st.markdown("#### 1. 地図をクリックして場所を指定")
            if HAS_FOLIUM:
                # 地図表示
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                # 既存スピーカーを表示
                for s in st.session_state.speakers:
                    folium.Marker(
                        [s['lat'], s['lon']], 
                        popup=s['label'],
                        icon=folium.Icon(color="green", icon="volume-up")
                    ).add_to(m)
                
                # クリックイベント取得
                output = st_folium(m, width="100%", height=400, return_last_object_clicked=True)
                
                if output and output.get("last_clicked"):
                    st.session_state.clicked_coords = output["last_clicked"]
            else:
                st.warning("`folium` がインストールされていないため、マップクリック機能は無効です。手動で入力してください。")
        
        with c_add_R:
            st.markdown("#### 2. 詳細を入力して追加")
            with st.form("click_add_form"):
                # クリックされた座標があればそれを初期値に
                init_lat = st.session_state.clicked_coords["lat"] if st.session_state.clicked_coords else center_lat
                init_lon = st.session_state.clicked_coords["lng"] if st.session_state.clicked_coords else center_lon
                
                in_lat = st.number_input("緯度", value=float(init_lat), format="%.6f")
                in_lon = st.number_input("経度", value=float(init_lon), format="%.6f")
                in_label = st.text_input("名称", value="新規地点")
                in_dir = st.number_input("方向 (度)", 0, 360, 0)
                
                if st.form_submit_button("この地点に追加", type="primary"):
                    st.session_state.speakers.append({
                        "lat": in_lat, "lon": in_lon, "label": in_label, "direction": in_dir
                    })
                    st.success("追加しました！シミュレーションタブで確認してください。")
                    st.session_state.clicked_coords = None # リセット
                    # st.rerun() はフォーム内だと警告が出る場合があるので外に出すか、自然更新に任せる

    # 3. AIコンサルタント (タイムアウト対策済み)
    with tab_ai:
        c_ai_1, c_ai_2 = st.columns([1, 2])
        
        with c_ai_1:
            st.subheader("AI コンサルタント")
            st.info("Gemini 2.0 Flash (Exp) が地形とシミュレーションを分析します。")
            user_query = st.text_area("指示・条件", "死角を解消するための最適な場所を1つ提案して。", height=100)
            
            if st.button("🚀 AIに配置案を作成させる"):
                with st.spinner("AIが地形解析中... (最大2分程度かかる場合があります)"):
                    prompt = IntelligentPlanner.generate_gemini_prompt(
                        user_query, st.session_state.speakers, blind_spot, params["L0"]
                    )
                    # 堅牢なAPI呼び出し
                    response_text = call_gemini_api_robust(prompt)
                    
                    if response_text:
                        st.session_state.last_response = response_text
                        json_match = re.search(r"```json\s*({.*?})\s*```", response_text, re.DOTALL)
                        if json_match:
                            try:
                                import json
                                prop_data = json.loads(json_match.group(1))
                                st.session_state.proposals = [prop_data]
                                st.success("提案地点をマップに追加しました！")
                            except:
                                st.warning("座標自動抽出に失敗")
        
        with c_ai_2:
            st.subheader("AI 分析レポート")
            if "last_response" in st.session_state:
                st.markdown(st.session_state.last_response)
                if st.session_state.proposals:
                    p = st.session_state.proposals[0]
                    if st.button("この提案を採用して配置する"):
                        st.session_state.speakers.append(p)
                        st.session_state.proposals = []
                        st.session_state.last_response = ""
                        st.rerun()
            else:
                st.write("ここにAIからの分析結果が表示されます。")

if __name__ == "__main__":
    main()
