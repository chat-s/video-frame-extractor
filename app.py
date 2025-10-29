import streamlit as st
import cv2
import os
from pathlib import Path
import tempfile
from PIL import Image
import zipfile
import io
import math

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

def compute_time_points(duration: float, step_sec: float = 5.0,
                        include_first: bool = True, include_last: bool = True):
    """
    计算取帧的时间点：
    - 从0开始，每 step_sec 秒取1帧（0, step, 2*step, ...）
    - 始终包含首帧(0)与末帧(duration)（末帧会做微调避免越界）
    """
    if duration <= 0:
        return [0.0] if include_first else []

    times = set()

    if include_first:
        times.add(0.0)

    if step_sec > 0:
        # 取到 < duration 的点
        n = int(math.floor(duration / step_sec))
        # 例如 duration=10.19, step=5 -> n=2 -> 5,10
        for k in range(1, n + 1):
            t = k * step_sec
            # 若恰好等于 duration，则略过，由 include_last 负责
            if t < duration:
                times.add(float(t))

    if include_last:
        # 末帧时间点微调，避免直接等于duration导致frame_number越界
        eps = 1e-2
        last_t = max(duration - eps, 0.0)
        times.add(float(last_t))

    # 排序后返回
    return sorted(times)

def extract_frames(video_path, step_sec=5.0):
    """
    按固定时间间隔提取帧，并包含首帧与末帧。
    :param video_path: 视频文件路径
    :param step_sec: 时间间隔（秒），例如5秒
    :return: 提取的帧列表
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("无法打开视频文件")
        return frames

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps is None or fps <= 0 or total_frames <= 0:
        st.error("视频元数据异常（FPS或总帧数无效）")
        cap.release()
        return frames

    duration = total_frames / fps

    st.info(f"📹 视频信息：时长 {duration:.2f} 秒 | 分辨率 {width}×{height} | 帧率 {fps:.2f} fps")

    # 计算时间点：0、step、2*step、...、最后一帧（微调）
    time_points = compute_time_points(duration, step_sec=step_sec, include_first=True, include_last=True)

    # 根据时间点提取帧
    for idx, t in enumerate(time_points, start=1):
        # 计算对应帧号并夹紧到 [0, total_frames-1]
        frame_number = int(t * fps)
        frame_number = min(max(frame_number, 0), total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            minutes = int(t) // 60
            seconds = int(t) % 60
            milliseconds = int((t % 1) * 100)
            frames.append({
                'frame': frame_rgb,
                'time': t,
                'label': f'第{idx}帧 ({minutes}:{seconds:02d}:{milliseconds:02d})'
            })

    cap.release()
    return frames

def save_frames_to_folder(frames, output_folder):
    """
    保存帧到文件夹
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    saved_files = []
    for idx, frame_data in enumerate(frames):
        filename = f"frame_{idx + 1}_{frame_data['time']:.2f}s.png"
        filepath = os.path.join(output_folder, filename)
        img = Image.fromarray(frame_data['frame'])
        img.save(filepath)
        saved_files.append(filepath)
    return saved_files

def create_zip_file(frames):
    """
    创建包含所有帧的ZIP文件（内存形式）
    """
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

# 主界面
st.title("🎬 视频帧提取工具")
st.markdown("上传视频，按固定时间间隔提取帧（默认每 5 秒一帧），并始终包含首帧与末帧。")

# 创建两列布局
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 上传视频")
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
        help="支持常见视频格式"
    )

    # 若你想固定为5秒，不让用户改，把下面 slider 删掉，直接在 extract_frames(video_path, step_sec=5.0)
    step_sec = st.slider(
        "时间间隔（秒）",
        min_value=1,
        max_value=60,
        value=5,
        help="每隔多少秒提取一帧（始终包含首帧与末帧）"
    )

    output_folder = st.text_input(
        "保存文件夹",
        value="extracted_frames",
        help="提取的帧将保存到此文件夹"
    )

with col2:
    st.subheader("📊 提取结果")

    if uploaded_file is not None:
        # 保存上传的视频到临时文件（保留原扩展名更稳妥）
        suffix = os.path.splitext(uploaded_file.name)[-1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        try:
            # 提取帧
            with st.spinner('正在提取视频帧...'):
                frames = extract_frames(video_path, step_sec=step_sec)

            if frames:
                st.success(f"✅ 成功提取 {len(frames)} 帧")
                # ZIP打包下载
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

                # 使用3列网格显示
                cols_per_row = 3
                for i in range(0, len(frames), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(frames):
                            with cols[j]:
                                frame_data = frames[i + j]
                                st.image(
                                    frame_data['frame'],
                                    caption=frame_data['label'],
                                    use_container_width=True
                                )
                                # 单张下载
                                img = Image.fromarray(frame_data['frame'])
                                img_buffer = io.BytesIO()
                                img.save(img_buffer, format='PNG')
                                img_buffer.seek(0)

                                st.download_button(
                                    label=f"💾 下载第{i+j+1}帧",
                                    data=img_buffer,
                                    file_name=f"frame_{i+j+1}_{frame_data['time']:.2f}s.png",
                                    mime="image/png",
                                    key=f"download_frame_{i+j+1}"
                                )
            else:
                st.warning("没有成功提取到帧，请检查视频文件。")
        finally:
            # 清理临时文件
            try:
                os.unlink(video_path)
            except Exception:
                pass
