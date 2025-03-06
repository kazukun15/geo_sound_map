import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import numpy as np
import pandas as pd
import math
import io
import branca.colormap as cm
import requests

# ------------------------------------------------------------------
# 定数／設定（APIキー、モデル）
# ------------------------------------------------------------------
API_KEY = st.secrets["general"]["api_key"]  # secrets.tomlで [general] api_key="..." を設定
MODEL_NAME = "gemini-2.0-flash"

# ----------------------------------------------------------------
# 省略: parse_direction, load_csv, export_csv, compute_sound_grid,
#       calculate_heatmap, calculate_objective, optimize_speaker_placement
#       generate_gemini_prompt, call_gemini_api
#       (既存のコードをそのまま使ってください)
# ----------------------------------------------------------------

def main():
    st.set_page_config(page_title="防災スピーカー音圧可視化マップ", layout="wide")
    st.title("防災スピーカー音圧可視化マップ")
    
    # --- セッションステート初期化（省略） ---
    if "map_center" not in st.session_state:
        st.session_state.map_center = [34.25741795269067, 133.20450105700033]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 14
    if "speakers" not in st.session_state:
        st.session_state.speakers = [[34.25741795269067, 133.20450105700033, [0.0, 90.0]]]
    if "measurements" not in st.session_state:
        st.session_state.measurements = []
    if "heatmap_data" not in st.session_state:
        st.session_state.heatmap_data = None
    if "L0" not in st.session_state:
        st.session_state.L0 = 80
    if "r_max" not in st.session_state:
        st.session_state.r_max = 500

    # --- サイドバーUI（CSVアップロード、スピーカー操作、Gemini API呼び出しなど） ---
    with st.sidebar:
        st.header("操作パネル")
        
        # （CSVアップロードやスピーカー追加・削除、パラメータ調整など既存の機能をここに記述）
        # ...
        
        # Gemini API 呼び出し
        st.subheader("Gemini API 呼び出し")
        gemini_query = st.text_input("Gemini に問い合わせる内容")
        if st.button("Gemini API を実行"):
            # プロンプト生成
            full_prompt = generate_gemini_prompt(gemini_query)
            # 実行結果をセッションに保存
            result = call_gemini_api(full_prompt)
            st.session_state.gemini_result = result
            st.success("Gemini API の実行が完了しました")

    # --- メインパネル（地図、ヒートマップ表示など） ---
    col1, col2 = st.columns([3, 1])
    with col1:
        # （ヒートマップ計算や Folium マップ表示など）
        # ...
        pass
    
    with col2:
        # （CSV ダウンロードなど）
        # ...
        pass
    
    # --- Gemini API の回答表示 ---
    st.markdown("---")
    st.subheader("Gemini API の回答（説明 & JSON）")
    
    if "gemini_result" in st.session_state:
        result = st.session_state.gemini_result
        
        # まず「説明」に相当するキーを探す（"explanation", "description" など）
        explanation_text = None
        # 優先的に探したいキーのリスト
        explanation_keys = ["explanation", "description"]
        
        # 1. トップレベルで探す
        for key in explanation_keys:
            if key in result:
                explanation_text = result[key]
                break
        
        # 2. candidates の中に探す
        if not explanation_text and "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            for key in explanation_keys:
                if key in candidate:
                    explanation_text = candidate[key]
                    break
        
        # -- 説明部分を表示 --
        if explanation_text:
            st.markdown("#### 説明部分")
            st.write(explanation_text)
        else:
            st.warning("説明部分を見つけられませんでした。")
        
        # -- JSON全体を表示 --
        st.markdown("#### JSON全体")
        st.json(result)
    else:
        st.info("Gemini API の回答はまだありません。")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"予期しないエラーが発生しました: {e}")
