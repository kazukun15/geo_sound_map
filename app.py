import streamlit as st
import pydeck as pdk
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# ここではシンプルに音圧計算のロジックだけ示す
def compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon):
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

def main():
    st.title("音圧分布を3Dカラムで可視化（Pydeck ColumnLayer）")
    
    # 例: スピーカー情報（[lat, lon, label]）
    # 実際には CSV や GeminiAPI 等で読み込む想定
    speakers = [
        [34.2574, 133.2045, "スピーカーA"],
        [34.2580, 133.2050, "スピーカーB"],
    ]
    
    # 音圧計算パラメータ
    L0 = 80
    r_max = 500
    
    # 地図の中心とグリッド生成
    map_center = [34.2574, 133.2045]
    lat_min = map_center[0] - 0.01
    lat_max = map_center[0] + 0.01
    lon_min = map_center[1] - 0.01
    lon_max = map_center[1] + 0.01
    
    Nx, Ny = 50, 50  # グリッド解像度（適宜調整）
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(lat_min, lat_max, Nx),
        np.linspace(lon_min, lon_max, Ny)
    )
    grid_lat = grid_lat.T  # shapeを (Nx, Ny)に合わせる
    grid_lon = grid_lon.T
    
    # 音圧計算
    sound_grid = compute_sound_grid(speakers, L0, r_max, grid_lat, grid_lon)
    
    # DataFrame へ格納
    data_list = []
    for i in range(Nx):
        for j in range(Ny):
            val = sound_grid[i, j]
            if not np.isnan(val):
                data_list.append({
                    "lat": grid_lat[i, j],
                    "lon": grid_lon[i, j],
                    "value": val
                })
    df = pd.DataFrame(data_list)
    
    # ColumnLayer 用に、音圧(dB)をそのまま高さにするとスケールが大きい or 小さい場合あり
    # 適宜スケールを調整
    # 例: dB 値から 3D高さに変換する簡易スケール
    df["elevation"] = (df["value"] - (L0 - 40)) * 10.0  # (dB-40) * 10 など
    
    # ColumnLayer
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["lon", "lat"],
        get_elevation="elevation",
        elevation_scale=1,         # 全体スケールを掛ける
        radius=30,                 # カラムの半径(メートル)
        get_fill_color=[255, 255, 0],  # カラー(R,G,B)
        pickable=True,
        auto_highlight=True,
    )
    
    # スピーカー表示用の ScatterplotLayer
    # 既存スピーカーを青アイコン、新設スピーカーを赤アイコン など実装も可能
    spk_df = pd.DataFrame(speakers, columns=["lat", "lon", "label"])
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=spk_df,
        get_position=["lon", "lat"],
        get_radius=100,
        get_fill_color=[0, 0, 255],  # 青
        pickable=True,
        auto_highlight=True,
    )
    
    # デッキ設定
    view_state = pdk.ViewState(
        latitude=map_center[0],
        longitude=map_center[1],
        zoom=14,
        pitch=40,
        bearing=0
    )
    deck = pdk.Deck(
        layers=[column_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"text": "値: {value}\nスピーカー: {label}"}
    )
    
    st.pydeck_chart(deck)
    
    # デバッグ
    st.write("Data size:", len(df))
    st.write("Example data:", df.head())

if __name__ == "__main__":
    main()
