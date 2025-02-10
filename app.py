import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import branca.colormap as cm
import json
from skimage import measure
import math
import csv
from geopy.geocoders import Nominatim

# -------------------------------------------------------------
# 1) セッションステートの初期化 (初回のみ)
# -------------------------------------------------------------
if "map_center" not in st.session_state:
    # 初期位置 (例: 下弓削・弓削総合庁舎)
    st.session_state.map_center = [34.25741795269067, 133.20450105700033]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 17
if "speakers" not in st.session_state:
    # 初期スピーカー（ホーン2本とも北向き）
    st.session_state.speakers = [
        [34.25741795269067, 133.20450105700033, [0.0, 0.0]]
    ]
if "measurements" not in st.session_state:
    st.session_state.measurements = []
if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None
if "iso_60" not in st.session_state:
    st.session_state.iso_60 = []
if "iso_80" not in st.session_state:
    st.session_state.iso_80 = []
if "searched_address" not in st.session_state:
    st.session_state.searched_address = None

# 前回のL0 / r_max
if "prev_l0" not in st.session_state:
    st.session_state.prev_l0 = None
if "prev_r_max" not in st.session_state:
    st.session_state.prev_r_max = None

# -------------------------------------------------------------
# 2) 方向を float角度に変換する関数
# -------------------------------------------------------------
def parse_direction_to_degrees(dir_str):
    dir_str = dir_str.strip().upper()
    mapping = {
        "N": 0, "NORTH": 0,
        "NE": 45,
        "E": 90, "EAST": 90,
        "SE": 135,
        "S": 180, "SOUTH": 180,
        "SW": 225,
        "W": 270, "WEST": 270,
        "NW": 315
    }
    if dir_str in mapping:
        return mapping[dir_str]
    else:
        return float(dir_str)

def fix_speakers_dir(speakers):
    """
    JSONインポート後などに呼び出し、文字列があればfloat化する（備考付きの場合にも対応）
    """
    fixed = []
    for spk in speakers:
        if len(spk) == 3:
            lat, lon, dir_list = spk
            remarks = ""
        elif len(spk) >= 4:
            lat, lon, dir_list, remarks = spk[0], spk[1], spk[2], spk[3]
        else:
            continue
        dir_fixed = []
        for d in dir_list:
            if isinstance(d, (int, float)):
                dir_fixed.append(d)
            else:
                dir_fixed.append(parse_direction_to_degrees(d))
        entry = [lat, lon, dir_fixed]
        if remarks:
            entry.append(remarks)
        fixed.append(entry)
    return fixed

# -------------------------------------------------------------
# 3) サイドバー (スピーカー追加/削除, 音圧パラメータ, プロジェクト入出力, 住所検索)
# -------------------------------------------------------------
st.sidebar.header("操作パネル")

# --- テキスト入力によるスピーカー追加
st.sidebar.subheader("スピーカー追加 (テキスト入力)")
new_speaker_input = st.sidebar.text_input(
    "緯度,経度,ホーン1向き,ホーン2向き\n例: 34.2579,133.2072,N,SW"
)
if st.sidebar.button("スピーカーを追加 (テキスト入力)"):
    try:
        lat_str, lon_str, d1_str, d2_str = new_speaker_input.split(",")
        lat_val = float(lat_str.strip())
        lon_val = float(lon_str.strip())
        dir1_deg = parse_direction_to_degrees(d1_str)
        dir2_deg = parse_direction_to_degrees(d2_str)
        st.session_state.speakers.append([lat_val, lon_val, [dir1_deg, dir2_deg]])
        st.session_state.heatmap_data = None
        st.sidebar.success(f"追加: lat={lat_val}, lon={lon_val}, horns=({d1_str},{d2_str})")
    except Exception as e:
        st.sidebar.error(f"形式が正しくありません (例: 34.2579,133.2072,N,SW)\nエラー: {e}")

# --- CSVによるスピーカー一括追加（ヘッダーなし）
# 各行は「緯度,経度,方向,備考」の順番になっているものとする
st.sidebar.subheader("スピーカーCSVインポート（ヘッダーなし）")
uploaded_speaker_csv = st.sidebar.file_uploader("CSVファイルをアップロード（ヘッダーなし）", type=["csv"], key="speaker_csv")
if uploaded_speaker_csv is not None:
    try:
        # BOM除去のため 'utf-8-sig' を指定
        decoded = uploaded_speaker_csv.read().decode('utf-8-sig').splitlines()
        reader = csv.reader(decoded)
        count = 0
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                lat_val = float(row[0].strip())
                lon_val = float(row[1].strip())
                direction_field = row[2].strip()
                if "," in direction_field:
                    directions = [d.strip() for d in direction_field.split(",")]
                    horn_directions = [parse_direction_to_degrees(d) for d in directions]
                else:
                    horn_directions = [parse_direction_to_degrees(direction_field)] * 2
                remarks = row[3].strip() if len(row) >= 4 else ""
                st.session_state.speakers.append([lat_val, lon_val, horn_directions, remarks])
                count += 1
            except Exception as row_e:
                st.sidebar.warning(f"行の読み込みに失敗しました: {row} エラー: {row_e}")
        st.session_state.heatmap_data = None
        st.sidebar.success(f"CSVから{count}件のスピーカーを追加しました。")
        # CSVアップロード後、ウィジェットをリセットするためキーを削除
        if "speaker_csv" in st.session_state:
            del st.session_state["speaker_csv"]
    except Exception as e:
        st.sidebar.error(f"CSVインポートに失敗しました: {e}")

# --- 住所検索機能
st.sidebar.subheader("住所検索")
address = st.sidebar.text_input("住所を入力してください", key="address_search")
if st.sidebar.button("検索", key="address_button"):
    if address.strip() == "":
        st.sidebar.error("住所を入力してください")
    else:
        try:
            geolocator = Nominatim(user_agent="my_app")
            location = geolocator.geocode(address)
            if location is None:
                st.sidebar.error("住所が見つかりませんでした")
            else:
                st.session_state.map_center = [location.latitude, location.longitude]
                # dBを計算
                db_value = calculate_single_point_db(st.session_state.speakers, L0, r_max, location.latitude, location.longitude)
                if db_value is None:
                    dB_text = "範囲外"
                else:
                    dB_text = f"{db_value:.2f} dB"
                st.session_state.searched_address = {
                    "address": address,
                    "lat": location.latitude,
                    "lon": location.longitude,
                    "db": dB_text
                }
                st.sidebar.success(f"{address} の位置: ({location.latitude:.6f}, {location.longitude:.6f}) dB: {dB_text}")
        except Exception as ge:
            st.sidebar.error(f"住所検索に失敗しました: {ge}")

# --- スピーカー削除
if st.session_state.speakers:
    st.sidebar.subheader("スピーカー削除")
    spk_index = st.sidebar.selectbox(
        "削除するスピーカー",
        range(len(st.session_state.speakers)),
        format_func=lambda i: f"{i+1}: {st.session_state.speakers[i]}"
    )
    if st.sidebar.button("削除"):
        removed = st.session_state.speakers.pop(spk_index)
        st.sidebar.warning(f"削除: {removed}")
        st.session_state.heatmap_data = None

# --- 音圧パラメータ
L0 = st.sidebar.slider("初期音圧レベル (dB)", 50, 100, 80)
r_max = st.sidebar.slider("最大伝播距離 (m)", 100, 2000, 500)
if st.session_state.prev_l0 is None:
    st.session_state.prev_l0 = L0
if st.session_state.prev_r_max is None:
    st.session_state.prev_r_max = r_max
if (L0 != st.session_state.prev_l0) or (r_max != st.session_state.prev_r_max):
    st.session_state.heatmap_data = None
    st.session_state.prev_l0 = L0
    st.session_state.prev_r_max = r_max

# --- プロジェクトの入出力 (JSON)
st.sidebar.subheader("プロジェクトのインポート/エクスポート")
uploaded_project = st.sidebar.file_uploader("JSONをインポート", type=["json"])
if uploaded_project is not None:
    try:
        project_data = json.load(uploaded_project)
        st.session_state.map_center = project_data["map_center"]
        st.session_state.speakers = fix_speakers_dir(project_data["speakers"])
        st.session_state.measurements = project_data["measurements"]
        st.session_state.heatmap_data = None
        st.sidebar.success("インポート完了")
    except Exception as e:
        st.sidebar.error(f"インポート失敗: {e}")
if st.sidebar.button("プロジェクトをエクスポート"):
    save_data = {
        "map_center": st.session_state.map_center,
        "speakers": st.session_state.speakers,
        "measurements": st.session_state.measurements
    }
    proj_json = json.dumps(save_data, ensure_ascii=False, indent=2)
    st.sidebar.download_button(
        "ダウンロード",
        proj_json.encode("utf-8"),
        file_name="project_data.json",
        mime="application/json"
    )

# -------------------------------------------------------------
# 4) 方位計算 & 指向性モデル
# -------------------------------------------------------------
def calc_bearing_deg(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    diff_lon_rad = math.radians(lon2 - lon1)
    x = math.sin(diff_lon_rad) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad)
         - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(diff_lon_rad))
    bearing = math.atan2(x, y)
    bearing_deg = math.degrees(bearing)
    return (bearing_deg + 360) % 360

def directivity_factor(bearing_spk_to_point, horn_angle, half_angle=60):
    diff = abs((bearing_spk_to_point - horn_angle + 180) % 360 - 180)
    if diff <= half_angle:
        return 1.0
    else:
        return 0.001  # -30dB

# -------------------------------------------------------------
# 5) (A) 複数スピーカー & 2ホーン 合成 → ヒートマップ & 等高線
# -------------------------------------------------------------
def calculate_heatmap_and_contours(speakers, L0, r_max, grid_lat, grid_lon):
    Nx, Ny = grid_lat.shape
    power_sum = np.zeros((Nx, Ny), dtype=float)
    in_range_mask = np.zeros((Nx, Ny), dtype=bool)
    for spk in speakers:
        if len(spk) >= 3:
            s_lat, s_lon, dir_list = spk[0], spk[1], spk[2]
        else:
            continue
        for i in range(Nx):
            for j in range(Ny):
                r = math.sqrt((grid_lat[i, j] - s_lat)**2 + (grid_lon[i, j] - s_lon)**2) * 111320
                if r == 0:
                    r = 1
                bearing_deg = calc_bearing_deg(s_lat, s_lon, grid_lat[i, j], grid_lon[i, j])
                power_spk = 0.0
                for horn_angle in dir_list:
                    df = directivity_factor(bearing_deg, horn_angle, 60)
                    p = df * (10**((L0 - 20 * math.log10(r)) / 10))
                    power_spk += p
                if power_spk > 0:
                    power_sum[i, j] += power_spk
                    if r <= r_max:
                        in_range_mask[i, j] = True
    sound_grid = 10 * np.log10(power_sum, where=(power_sum > 0), out=np.full_like(power_sum, np.nan))
    sound_grid[~in_range_mask] = np.nan
    heat_data = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                heat_data.append([grid_lat[i, j], grid_lon[i, j], val])
    iso_60 = []
    iso_80 = []
    if Nx >= 2 and Ny >= 2:
        cgrid = np.where(np.isnan(sound_grid), -9999, sound_grid)
        cs60 = measure.find_contours(cgrid, 60)
        for contour in cs60:
            coords = []
            for y, x in contour:
                iy, ix = int(y), int(x)
                if 0 <= iy < Nx and 0 <= ix < Ny:
                    coords.append((grid_lat[iy, ix], grid_lon[iy, ix]))
            if len(coords) > 1:
                iso_60.append(coords)
        cs80 = measure.find_contours(cgrid, 80)
        for contour in cs80:
            coords = []
            for y, x in contour:
                iy, ix = int(y), int(x)
                if 0 <= iy < Nx and 0 <= ix < Ny:
                    coords.append((grid_lat[iy, ix], grid_lon[iy, ix]))
            if len(coords) > 1:
                iso_80.append(coords)
    return heat_data, iso_60, iso_80

# -------------------------------------------------------------
# 5) (B) 単一点計算 (計測値との比較)
# -------------------------------------------------------------
def calculate_single_point_db(speakers, L0, r_max, lat, lon):
    total_power = 0.0
    in_range = False
    for spk in speakers:
        if len(spk) >= 3:
            s_lat, s_lon, dir_list = spk[0], spk[1], spk[2]
        else:
            continue
        r = math.sqrt((lat - s_lat)**2 + (lon - s_lon)**2) * 111320
        if r == 0:
            r = 1
        bearing_deg = calc_bearing_deg(s_lat, s_lon, lat, lon)
        power_spk = 0.0
        for horn_angle in dir_list:
            df = directivity_factor(bearing_deg, horn_angle, 60)
            p = df * (10**((L0 - 20 * math.log10(r)) / 10))
            power_spk += p
        if power_spk > 0:
            total_power += power_spk
            if r <= r_max:
                in_range = True
    if (total_power <= 0) or (not in_range):
        return None
    return 10 * math.log10(total_power)

# -------------------------------------------------------------
# 6) メッシュ範囲 & ヒートマップ再計算
# -------------------------------------------------------------
def get_speaker_bounds(speakers, margin=0.01):
    if not speakers:
        return (
            st.session_state.map_center[0] - margin,
            st.session_state.map_center[0] + margin,
            st.session_state.map_center[1] - margin,
            st.session_state.map_center[1] + margin
        )
    lat_list = [s[0] for s in speakers]
    lon_list = [s[1] for s in speakers]
    return (min(lat_list) - margin, max(lat_list) + margin,
            min(lon_list) - margin, max(lon_list) + margin)

lat_min, lat_max, lon_min, lon_max = get_speaker_bounds(st.session_state.speakers, margin=0.01)
N = 100
grid_lat, grid_lon = np.meshgrid(
    np.linspace(lat_min, lat_max, N),
    np.linspace(lon_min, lon_max, N)
)

if st.session_state.heatmap_data is None:
    if len(st.session_state.speakers) == 0:
        st.session_state.heatmap_data = []
        st.session_state.iso_60 = []
        st.session_state.iso_80 = []
    else:
        hd, iso60, iso80 = calculate_heatmap_and_contours(st.session_state.speakers, L0, r_max, grid_lat, grid_lon)
        st.session_state.heatmap_data = hd
        st.session_state.iso_60 = iso60
        st.session_state.iso_80 = iso80

# -------------------------------------------------------------
# 7) Folium地図表示
#    ユーザー操作（ズーム/パン/クリック）情報取得とセッションステートの更新
# -------------------------------------------------------------
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=st.session_state.map_zoom
)

# --- スピーカーのマーカー表示（備考がある場合はポップアップに表示）
for spk in st.session_state.speakers:
    if len(spk) >= 3:
        lat_s = spk[0]
        lon_s = spk[1]
        dir_list = spk[2]
        remarks = spk[3] if len(spk) >= 4 else ""
        if remarks:
            popup_str = f"施設名: {remarks}\nSpk: ({lat_s:.6f},{lon_s:.6f})\n向き: {dir_list}"
        else:
            popup_str = f"Spk: ({lat_s:.6f},{lon_s:.6f})\n向き: {dir_list}"
        folium.Marker(location=[lat_s, lon_s], popup=popup_str).add_to(m)

# --- 住所検索結果のピンを追加（ある場合）
if st.session_state.searched_address:
    addr = st.session_state.searched_address
    folium.Marker(
        location=[addr["lat"], addr["lon"]],
        icon=folium.Icon(color="red", icon="info-sign"),
        popup=f"住所: {addr['address']}<br>dB: {addr['db']}"
    ).add_to(m)

# --- st_folium の呼び出しで、"last_clicked" も取得
st_data = st_folium(m, width=800, height=600, returned_objects=["center", "zoom", "last_clicked"])
if st_data and isinstance(st_data, dict):
    new_center = st_data.get("center")
    new_zoom = st_data.get("zoom")
    if new_center:
        if isinstance(new_center, dict):
            lat = new_center.get("lat", st.session_state.map_center[0])
            lng = new_center.get("lng", st.session_state.map_center[1])
            st.session_state.map_center = [lat, lng]
        elif isinstance(new_center, list) and len(new_center) == 2:
            st.session_state.map_center = new_center
    if new_zoom is not None:
        st.session_state.map_zoom = new_zoom

    # --- クリック位置の処理
    if "last_clicked" in st_data and st_data["last_clicked"]:
        click_lat = st_data["last_clicked"]["lat"]
        click_lon = st_data["last_clicked"]["lng"]
        sound_val = calculate_single_point_db(st.session_state.speakers, L0, r_max, click_lat, click_lon)
        if sound_val is None:
            sound_text = "範囲外"
        else:
            sound_text = f"{sound_val:.2f} dB"
        st.sidebar.info(f"クリック位置 ({click_lat:.6f}, {click_lon:.6f}) の音圧: {sound_text}")

# --- ズームレベルに応じた HeatMap パラメータの動的調整
current_zoom = st.session_state.map_zoom
adjusted_radius = max(5, 15 * (17 / current_zoom))
adjusted_blur   = max(10, 20 * (17 / current_zoom))
HeatMap(
    st.session_state.heatmap_data,
    min_opacity=0.4,
    max_val=L0,
    radius=adjusted_radius,
    blur=adjusted_blur
).add_to(m)
for coords in st.session_state.iso_60:
    folium.PolyLine(coords, color="blue", weight=2, tooltip="60dB 等高線").add_to(m)
for coords in st.session_state.iso_80:
    folium.PolyLine(coords, color="red", weight=2, tooltip="80dB 等高線").add_to(m)
colormap = cm.LinearColormap(["blue", "green", "yellow", "red"], vmin=L0 - 40, vmax=L0)
colormap.caption = "音圧レベル (dB)"
m.add_child(colormap)

# -------------------------------------------------------------
# 8) 計測値入力 & 一覧
# -------------------------------------------------------------
st.subheader("計測値の直接入力 (lat, lon, 実測dB)")
inp = st.text_input("例: 34.2579,133.2072,75.0")
if st.button("計測値を追加"):
    if inp.strip():
        try:
            lat_s, lon_s, db_s = inp.split(",")
            lat_v = float(lat_s.strip())
            lon_v = float(lon_s.strip())
            meas_db = float(db_s.strip())
            st.session_state.measurements.append([lat_v, lon_v, meas_db])
            st.success(f"追加: ({lat_v},{lon_v}), 実測={meas_db}dB")
        except Exception as e:
            st.warning(f"形式が正しくありません (例: 34.2579,133.2072,75.0)\nエラー: {e}")
    else:
        st.warning("入力が空です")
st.header("計測結果一覧")
if not st.session_state.measurements:
    st.write("まだありません。")
else:
    for (lat_m, lon_m, m_db) in st.session_state.measurements:
        calc_db = calculate_single_point_db(st.session_state.speakers, L0, r_max, lat_m, lon_m)
        if calc_db is None:
            st.write(f"({lat_m:.6f},{lon_m:.6f}): 実測={m_db:.2f}dB, 計算=範囲外")
        else:
            st.write(f"({lat_m:.6f},{lon_m:.6f}): 実測={m_db:.2f}dB, 計算={calc_db:.2f}dB")
