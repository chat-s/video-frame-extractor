import streamlit as st
import cv2
import os
from pathlib import Path
import tempfile
from PIL import Image
import zipfile
import io
import math
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="视频帧提取工具",
    page_icon="🎬",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #4338CA; }
    .frame-container {
        border: 2px solid #E5E7EB;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- 计算时间点 ----------
def compute_time_points(duration: float, step_sec: float = 5.0,
                        include_first: bool = True, include_last: bool = True):
    if duration <= 0:
        return [0.0] if include_first else []
    times = set()
    if include_first:
        times.add(0.0)
    if step_sec > 0:
        n = int(math.floor(duration / step_sec))
        for k in range(1, n + 1):
            t = k * step_sec
            if t < duration:
                times.add(float(t))
    if include_last:
        eps = 1e-2
        times.add(float(max(duration - eps, 0.0)))
    return sorted(times)

# ---------- 清晰度评估 ----------
def frame_sharpness(gray: np.ndarray) -> float:
    # 拉普拉斯方差，值越大越清晰
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def read_frame_by_number(cap: cv2.VideoCapture, frame_num: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = frame_sharpness(gray)
    return rgb, float(score)

def format_label(idx: int, t: float):
    minutes = int(t) // 60
    seconds = int(t) % 60
    ms = int((t % 1) * 100)
    return f'第{idx}帧 ({minutes}:{seconds:02d}:{ms:02d})'

# ---------- 提取（每段采样多帧，选最清晰） ----------
def extract_frames_multi(video_path, step_sec=5.0, samples_per_segment=5, window_sec=0.6,
                         strategy="best"):  # strategy: "best" 或 "topk"
    """
    - 对每个目标时间点t，在[t-window/2, t+window/2]内均匀采样若干帧
    - 计算清晰度评分，按策略保留
    """
    results = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("无法打开视频文件")
        return results

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps is None or fps <= 0 or total_frames <= 0:
        st.error("视频元数据异常（FPS或总帧数无效）")
        cap.release()
        return results

    duration = total_frames / fps
    st.info(f"📹 视频信息：时长 {duration:.2f} 秒 | 分辨率 {width}×{height} | 帧率 {fps:.2f} fps")

    time_points = compute_time_points(duration, step_sec=step_sec, include_first=True, include_last=True)

    half_w = max(window_sec / 2.0, 0.0)
    global_index = 1

    for base_t in time_points:
        # 生成该段的候选时间点（包含base_t本身）
        if samples_per_segment <= 1:
            candidate_times = [base_t]
        else:
            start_t = max(base_t - half_w, 0.0)
            end_t = min(base_t + half_w, duration - 1e-2)
            if end_t < start_t:
                start_t, end_t = end_t, end_t  # 退化为一个点
            # 均匀采样 samples_per_segment 个候选时间
            if samples_per_segment == 1:
                candidate_times = [base_t]
            else:
                candidate_times = [
                    start_t + i * (end_t - start_t) / (samples_per_segment - 1)
                    for i in range(samples_per_segment)
                ]
            # 确保包含 base_t
            candidate_times.append(base_t)

        # 映射到帧号并去重
        candidate_frame_nums = sorted(set([
            min(max(int(round(t * fps)), 0), total_frames - 1) for t in candidate_times
        ]))

        # 读取候选帧并评分
        candidate_frames = []
        for fn in candidate_frame_nums:
            rgb, score = read_frame_by_number(cap, fn)
            if rgb is None:
                continue
            t_est = min(fn / fps, duration - 1e-2)
            candidate_frames.append((fn, t_est, rgb, score))

        if not candidate_frames:
            continue

        # 按清晰度排序（降序）
        candidate_frames.sort(key=lambda x: x[3], reverse=True)

        # 选择策略
        selected = []
        if strategy == "best":
            selected = candidate_frames[:1]
        else:  # "topk"
            k = min(samples_per_segment, len(candidate_frames))
            selected = candidate_frames[:k]

        # 写入结果
        for _, t_sel, rgb_sel, score_sel in selected:
            results.append({
                'frame': rgb_sel,
                'time': float(t_sel),
                'sharpness': float(score_sel),
                'label': format_label(global_index, float(t_sel))
            })
            global_index += 1

    cap.release()
    # 最终按时间排序（可选）
    results.sort(key=lambda x: x['time'])
    # 重新编号标签
    for i, item in enumerate(results, start=1):
        item['label'] = format_label(i, item['time'])
    return results

def create_zip_file(frames):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for idx, frame_data in enumerate(frames):
            filename = f"frame_{idx + 1}_{frame_data['time']:.2f}s.png"
            img = Image.fromarray(frame_data['frame'])
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(filename, img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# ---------------- UI ----------------
st.title("🎬 视频帧提取工具（抗模糊版）")
st.markdown("按固定时间间隔提帧，并在每段内**采样多帧选择更清晰**。")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 上传视频")
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
        help="支持常见视频格式"
    )

    step_sec = st.slider(
        "时间间隔（秒）",
        min_value=1, max_value=60, value=5,
        help="每隔多少秒提取一段（始终包含首帧与末帧）"
    )

    samples_per_segment = st.slider(
        "每段采样帧数",
        min_value=1, max_value=10, value=5,
        help="每个时间段内取多少候选帧，用来挑选更清晰的帧"
    )

    window_sec = st.slider(
        "采样窗口（秒）",
        min_value=0, max_value=2, value=1,
        help="围绕目标时间点的窗口宽度（例如 1 秒 = ±0.5 秒）"
    )

    strategy = st.selectbox(
        "选择策略",
        ["最清晰一张", "保留前K张"],
        help="每段只保留一张最佳，或保留按清晰度排序的前K张（K=每段采样帧数）"
    )
    strategy_key = "best" if strategy == "最清晰一张" else "topk"

with col2:
    st.subheader("📊 提取结果")

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[-1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        try:
            with st.spinner('正在提取视频帧...'):
                frames = extract_frames_multi(
                    video_path,
                    step_sec=step_sec,
                    samples_per_segment=samples_per_segment,
                    window_sec=float(window_sec),
                    strategy=strategy_key
                )

            if frames:
                st.success(f"✅ 成功提取 {len(frames)} 帧")
                # 显示ZIP下载
                zip_bytes = create_zip_file(frames)
                st.download_button(
                    label="📦 下载全部帧（ZIP）",
                    data=zip_bytes,
                    file_name="extracted_frames.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                st.markdown("---")
                st.subheader("🖼️ 预览提取的帧")

                cols_per_row = 3
                for i in range(0, len(frames), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(frames):
                            with cols[j]:
                                f = frames[i + j]
                                st.image(f['frame'], caption=f"{f['label']} | 清晰度{f['sharpness']:.0f}", use_container_width=True)
                                img = Image.fromarray(f['frame'])
                                buf = io.BytesIO()
                                img.save(buf, format='PNG')
                                buf.seek(0)
                                st.download_button(
                                    label=f"💾 下载第{i+j+1}帧",
                                    data=buf,
                                    file_name=f"frame_{i+j+1}_{f['time']:.2f}s.png",
                                    mime="image/png",
                                    key=f"download_frame_{i+j+1}"
                                )
            else:
                st.warning("没有成功提取到帧，请检查视频文件或参数设置。")
        finally:
            try:
                os.unlink(video_path)
            except Exception:
                pass
