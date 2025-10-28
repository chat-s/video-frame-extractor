import streamlit as st
import cv2
import os
from pathlib import Path
import tempfile
from PIL import Image
import zipfile
import io

# 设置页面配置
st.set_page_config(
    page_title="视频帧提取工具",
    page_icon="🎬",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4338CA;
    }
    .frame-container {
        border: 2px solid #E5E7EB;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


def extract_frames(video_path, interval=5, num_frames=3):
    """
    从视频中提取指定数量的帧
    :param video_path: 视频文件路径
    :param interval: 提取间隔（秒）
    :param num_frames: 要提取的帧数
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
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"📹 视频信息：时长 {duration:.2f}秒 | 分辨率 {width}×{height} | 帧率 {fps:.2f}fps")

    # 计算均匀分布的时间点
    time_points = []

    # 提取第一帧
    time_points.append(0)

    # 均匀分布其他帧
    if num_frames > 1:
        step = duration / (num_frames - 1)
        for i in range(1, num_frames - 1):
            time_points.append(step * i)
        # 提取最后一帧
        time_points.append(duration - 0.01)

    # 根据时间点提取帧
    frame_count = 1
    for time in time_points:
        frame_number = int(time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            minutes = int(time) // 60
            seconds = int(time) % 60
            milliseconds = int((time % 1) * 100)
            frames.append({
                'frame': frame_rgb,
                'time': time,
                'label': f'第{frame_count}帧 ({minutes}:{seconds:02d}:{milliseconds:02d})'
            })
            frame_count += 1

    cap.release()
    return frames


def save_frames_to_folder(frames, output_folder):
    """
    保存帧到文件夹
    :param frames: 帧列表
    :param output_folder: 输出文件夹路径
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
    创建包含所有帧的ZIP文件
    :param frames: 帧列表
    :return: ZIP文件的字节流
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
st.markdown("上传视频，自动提取均匀分布的视频帧")

# 创建两列布局
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 上传视频")
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'],
        help="支持常见视频格式"
    )

    num_frames = st.slider(
        "要提取的帧数",
        min_value=1,
        max_value=10,
        value=3,
        help="从视频中均匀提取多少帧"
    )

    output_folder = st.text_input(
        "保存文件夹",
        value="extracted_frames",
        help="提取的帧将保存到此文件夹"
    )

with col2:
    st.subheader("📊 提取结果")

    if uploaded_file is not None:
        # 保存上传的视频到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # 提取帧
        with st.spinner('正在提取视频帧...'):
            frames = extract_frames(video_path, num_frames=num_frames)

        if frames:
            st.success(f"✅ 成功提取 {len(frames)} 帧")

            # 保存按钮
            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("💾 保存到文件夹"):
                    with st.spinner('正在保存...'):
                        saved_files = save_frames_to_folder(frames, output_folder)
                        st.success(f"已保存 {len(saved_files)} 张图片到 {output_folder} 文件夹")

            with col_btn2:
                zip_data = create_zip_file(frames)
                st.download_button(
                    label="📦 下载ZIP压缩包",
                    data=zip_data,
                    file_name="extracted_frames.zip",
                    mime="application/zip"
                )

            # 显示提取的帧
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

        # 清理临时文件
        os.unlink(video_path)

