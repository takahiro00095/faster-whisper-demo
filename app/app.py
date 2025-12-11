import streamlit as st
from faster_whisper import WhisperModel
import os
from pydub import AudioSegment  # 追加: 音声処理用ライブラリ

# 一時保存ディレクトリ
UPLOAD_DIR = "/tmp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def preprocess_audio(file_path):
    """
    音声をWhisper向けに最適化する関数
    1. ステレオ -> モノラル変換
    2. サンプリングレートを16kHzに変換
    3. 音量を正規化 (Normalize)
    """
    try:
        # 音声読み込み
        audio = AudioSegment.from_file(file_path)
        
        # 1. モノラル化 (Whisperはモノラルで処理するため)
        audio = audio.set_channels(1)
        
        # 2. 16kHzに変換 (Whisperのネイティブレート)
        audio = audio.set_frame_rate(16000)
        
        # 3. 音量正規化 (-20dBFSをターゲットにする)
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)
        
        # 処理済みファイル名を作成 (例: audio.mp3 -> audio_prep.wav)
        base, _ = os.path.splitext(file_path)
        new_path = base + "_prep.wav"
        
        # 書き出し
        normalized_audio.export(new_path, format="wav")
        return new_path, None
    except Exception as e:
        return None, f"前処理エラー: {e}"

st.title("🪶 FasterWhisper Demo")

# --- サイドバー設定 ---
with st.sidebar:
    st.header("モデル設定")
    model_size = st.selectbox(
        "モデルサイズ", 
        ["large-v3-turbo", "medium", "small", "base"], 
        index=0
    )
    compute_type = st.selectbox("計算タイプ", ["int8", "float16"], index=1)
    
    st.divider()
    
    st.header("オプション")
    use_preprocessing = st.checkbox("音声の前処理を行う", value=True, help="音量を均一化し、認識精度を高めます。時間が少しかかります。")
    use_vad = st.checkbox("VADフィルター (無音除去)", value=True)
    beam_size = st.slider("Beam Size", 1, 5, 5, help="1が最速。5が高精度。")

# --- メイン処理 ---
uploaded_file = st.file_uploader("音声ファイル (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if st.button("文字起こし開始"):
    if uploaded_file is None:
        st.error("ファイルをセットしてください")
    else:
        # 1. ファイル保存
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. 前処理 (オプション)
        final_file_path = file_path
        if use_preprocessing:
            with st.spinner("音声の前処理中（正規化・変換）..."):
                prep_path, error = preprocess_audio(file_path)
                if error:
                    st.warning(f"前処理をスキップしました: {error}")
                else:
                    final_file_path = prep_path
                    st.success("前処理完了: 音量を最適化しました")

        # 3. モデルロード
        try:
            st.info(f"モデル '{model_size}' をロード中...")
            model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        except Exception as e:
            st.error(f"モデルエラー: {e}")
            st.stop()

        st.info("解析中... (リアルタイム表示)")
        
        # 4. 推論実行
        # VADやBeamSizeなどのパラメータ設定
        segments, info = model.transcribe(
            final_file_path, 
            beam_size=beam_size, 
            vad_filter=use_vad,
            vad_parameters=dict(min_silence_duration_ms=500) if use_vad else None
        )
        
        st.success(f"検出言語: {info.language} (確率: {int(info.language_probability * 100)}%)")

        # 5. 結果表示 (高速化バージョン)
        full_text_list = []
        log_container = st.container()
        
        # ログコンテナの高さを指定（CSSハックなしの簡易版）
        with log_container:
             for segment in segments:
                # start = f"{segment.start:.1f}"
                # end = f"{segment.end:.1f}"
                line = f"{segment.text}"
                
                # 軽い表示メソッドを使用
                st.text(line) 
                full_text_list.append(line)

        # 最終結果出力
        output_text = "\n".join(full_text_list)
        st.divider()
        st.subheader("全結果 (コピー用)")
        st.text_area("結果", output_text, height=400)