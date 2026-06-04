'''
智能宠物品种识别与养护建议系统 - 主程序
基于 Streamlit 的 Web 交互界面
'''
import streamlit as st
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="宠物品种识别与养护系统",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- 自定义 CSS 样式 ----------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.5s;
    }
    .team-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        color: #1a1a1a;
    }
    .team-card h4 {
        color: #1a1a1a;
        font-weight: 600;
    }
    .team-card p {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)


# ---------- 延迟加载模型 ----------
@st.cache_resource
def load_classifier():
    '''加载品种识别模型（带缓存）'''
    from models.predict import PetBreedClassifier
    return PetBreedClassifier()


@st.cache_resource
def load_advisor(backend, model_name):
    '''加载 LLM 顾问（带缓存）'''
    from llm.advisor import PetCareAdvisor
    try:
        return PetCareAdvisor(backend=backend, model_name=model_name)
    except Exception as e:
        st.warning(f"LLM 初始化失败: {e}，将使用离线模式")
        from llm.advisor import PetCareAdvisor
        advisor = PetCareAdvisor(backend="fallback")
        advisor.backend = "fallback"
        return advisor


# ---------- 侧边栏 ----------
with st.sidebar:
    st.title("🐾 宠物智能识别")

    page = st.radio(
        "功能导航",
        ["🏠 首页", "📸 品种识别", "💬 养护顾问", "📊 品种对比", "🏥 健康咨询", "ℹ️ 关于项目"],
        label_visibility="collapsed"
    )

    st.divider()

    # LLM 配置
    st.subheader("⚙️ LLM 配置")
    llm_backend = st.selectbox(
        "大模型后端",
        ["ollama", "transformers", "fallback"],
        help="ollama: 本地Ollama服务 | transformers: HuggingFace模型 | fallback: 离线规则"
    )
    if llm_backend == "ollama":
        llm_model = st.text_input("模型名称", value="qwen2.5:1.5b")
    elif llm_backend == "transformers":
        llm_model = st.text_input("模型名称", value="Qwen/Qwen2.5-0.5B-Instruct")
    else:
        llm_model = "fallback"

    st.divider()
    


# ---------- 首页 ----------
if page == "🏠 首页":
    st.markdown('<div class="main-header"><h1>🐾 智能宠物品种识别与养护建议系统</h1><p>基于深度学习与大模型的宠物智能服务</p></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("可识别品种", "37 种", "猫狗品种")
    with col2:
        st.metric("模型准确率", "≥92%", "目标指标")
    with col3:
        st.metric("响应速度", "<1s", "单图推理")

    st.markdown("---")

    st.subheader("📋 系统功能概览")
    func_cols = st.columns(4)
    with func_cols[0]:
        st.info("**📸 品种识别**\n上传宠物照片，AI 自动识别品种，支持 37 种猫狗品种")
    with func_cols[1]:
        st.success("**💬 养护顾问**\n基于大模型生成个性化养护方案，涵盖饮食、健康、训练")
    with func_cols[2]:
        st.warning("**📊 品种对比**\n对比不同品种特点，辅助选宠决策")
    with func_cols[3]:
        st.error("**🏥 健康咨询**\n提供常见健康问题分析与应急建议")

    st.markdown("---")

    st.subheader("🔬 技术架构")
    st.markdown("""
    | 模块 | 技术栈 | 说明 |
    |------|--------|------|
    | 图像识别 | PyTorch + ResNet-18 | 迁移学习，37类猫狗品种分类 |
    | 大语言模型 | Ollama / Transformers | 本地部署，生成养护建议 |
    | 前端界面 | Streamlit | Python Web应用框架 |
    | 数据处理 | torchvision + PIL | 图像预处理与增强 |
    """)

    st.subheader("🚀 快速开始")
    st.code("双击 run.bat 一键启动  或  命令行执行: streamlit run app.py", language="bash")


# ---------- 品种识别 ----------
elif page == "📸 品种识别":
    st.markdown('<div class="main-header"><h2>📸 宠物品种智能识别</h2></div>', unsafe_allow_html=True)

    upload_col, result_col = st.columns([1, 1])

    with upload_col:
        st.subheader("上传宠物照片")
        uploaded_file = st.file_uploader(
            "支持 JPG、PNG 格式",
            type=["jpg", "jpeg", "png"],
            help="请上传清晰的宠物正面照片以获得最佳识别效果"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="待识别图片", use_container_width=True)

            if st.button("🔍 开始识别", type="primary", use_container_width=True):
                with st.spinner("AI 分析中..."):
                    try:
                        classifier = load_classifier()
                        result = classifier.predict(image)
                        st.session_state["last_result"] = result
                        st.session_state["show_result"] = True
                    except Exception as e:
                        st.error(f"识别失败: {e}")
                        st.info("提示：首次运行需下载预训练模型 (~45MB)")

    with result_col:
        st.subheader("识别结果")
        if st.session_state.get("show_result") and st.session_state.get("last_result"):
            result = st.session_state["last_result"]

            st.markdown(f"""
            <div class="result-card">
                <h3 style="color:#667eea; margin:0;">{result['breed_cn']}</h3>
                <p style="color:#666; margin:5px 0 0 0;">{result['breed']}</p>
                <div style="margin-top:15px;">
                    <span style="font-weight:bold;">置信度: {result['confidence']}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{result['confidence']}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top-5 结果柱状图
            top5 = result["top5"]
            if top5:
                st.subheader("Top-5 预测结果")
                fig, ax = plt.subplots(figsize=(8, 3))
                breeds = [r["breed_cn"][:6] for r in reversed(top5)]
                confs = [r["confidence"] for r in reversed(top5)]
                colors = ['#667eea' if i == len(top5)-1 else '#a0a0a0' for i in range(len(top5))]

                ax.barh(breeds, confs, color=colors, height=0.6)
                ax.set_xlabel("置信度 (%)")
                ax.set_xlim(0, 100)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                for i, (breed, conf) in enumerate(zip(breeds, confs)):
                    ax.text(conf + 1, i, f"{conf:.1f}%", va='center', fontsize=9)

                st.pyplot(fig)
                plt.close()


# ---------- 养护顾问 ----------
elif page == "💬 养护顾问":
    st.markdown('<div class="main-header"><h2>💬 智能养护顾问</h2></div>', unsafe_allow_html=True)

    if "last_result" not in st.session_state:
        st.warning("⚠️ 请先在「品种识别」页面识别宠物品种，或手动输入品种信息")
        manual_breed = st.text_input("手动输入品种名称", placeholder="例如: 金毛寻回犬、布偶猫")
        if manual_breed:
            st.session_state["manual_breed"] = manual_breed
    else:
        manual_breed = None

    breed_info = None
    if st.session_state.get("last_result"):
        breed_info = st.session_state["last_result"]
        st.success(f"当前品种: **{breed_info['breed_cn']}** ({breed_info['breed']}) - 置信度: {breed_info['confidence']}%")
    elif st.session_state.get("manual_breed"):
        breed_info = {"breed": st.session_state["manual_breed"], "breed_cn": st.session_state["manual_breed"], "confidence": 100}
        st.info(f"手动输入品种: **{st.session_state['manual_breed']}**")

    if breed_info:
        advice_type = st.selectbox(
            "选择建议类型",
            ["综合养护建议", "训练指导", "常见问题答疑"]
        )

        if st.button("✨ 生成养护建议", type="primary"):
            with st.spinner("大模型正在生成建议..."):
                try:
                    advisor = load_advisor(llm_backend, llm_model)
                    if advice_type == "综合养护建议":
                        st.write_stream(advisor.get_care_advice_stream(
                            breed_info["breed"], breed_info["breed_cn"]
                        ))
                    elif advice_type == "训练指导":
                        st.write_stream(advisor.training_advice_stream(
                            breed_info["breed"], breed_info["breed_cn"], "基础服从训练"
                        ))
                    else:
                        st.write_stream(advisor.health_advice_stream(
                            breed_info["breed_cn"], "日常健康注意事项"
                        ))
                except Exception as e:
                    st.error(f"生成失败: {e}")


# ---------- 品种对比 ----------
elif page == "📊 品种对比":
    st.markdown('<div class="main-header"><h2>📊 品种对比分析</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        breed_a = st.text_input("品种A", placeholder="例如: 拉布拉多", key="breed_a")
    with col2:
        breed_b = st.text_input("品种B", placeholder="例如: 金毛寻回犬", key="breed_b")

    if breed_a and breed_b:
        if st.button("🔍 开始对比", type="primary"):
            with st.spinner("大模型分析中..."):
                try:
                    advisor = load_advisor(llm_backend, llm_model)
                    st.write_stream(advisor.compare_breeds_stream(breed_a, breed_a, breed_b, breed_b))
                except Exception as e:
                    st.error(f"对比失败: {e}")


# ---------- 健康咨询 ----------
elif page == "🏥 健康咨询":
    st.markdown('<div class="main-header"><h2>🏥 宠物健康咨询</h2></div>', unsafe_allow_html=True)

    st.warning("⚠️ **免责声明**: AI 建议仅供参考，不能替代专业兽医诊断。紧急情况请立即就医！")

    health_breed = ""
    if st.session_state.get("last_result"):
        health_breed = st.session_state["last_result"]["breed_cn"]
    health_breed = st.text_input("宠物品种", value=health_breed, placeholder="请输入宠物品种")

    user_question = st.text_area(
        "描述您的疑问",
        placeholder="例如：我家猫咪最近食欲不振，精神也不好，可能是什么原因？应该怎么办？",
        height=120
    )

    if health_breed and user_question:
        if st.button("🏥 获取建议", type="primary"):
            with st.spinner("分析中..."):
                try:
                    advisor = load_advisor(llm_backend, llm_model)
                    st.write_stream(advisor.health_advice_stream(health_breed, user_question))
                except Exception as e:
                    st.error(f"请求失败: {e}")


# ---------- 关于项目 ----------
elif page == "ℹ️ 关于项目":
    st.markdown('<div class="main-header"><h2>ℹ️ 项目信息</h2></div>', unsafe_allow_html=True)

    st.subheader("📝 项目简介")
    st.markdown("""
    本项目基于深度学习和自然语言处理技术，
    构建一个实用的智能宠物品种识别与养护建议系统。系统能够自动识别宠物品种，
    并结合大语言模型生成个性化的养护建议，为宠物主人提供一站式智能服务。
    """)

    st.subheader("👥 贡献者")
    team_cols = st.columns(2)
    with team_cols[0]:
        st.markdown("""
        <div class="team-card">
            <h4>🐱 小小诺亚</h4>
            <p>职责：项目架构设计、模型开发与训练、前端开发</p>
        </div>
        """, unsafe_allow_html=True)
    with team_cols[1]:
        st.markdown("""
        <div class="team-card">
            <h4>🐶 Legend2333</h4>
            <p>职责：大模型集成、提示词工程、文档与测试</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🛠️ 技术栈")
    st.markdown("""
    - **深度学习框架**: PyTorch + torchvision
    - **模型架构**: ResNet-18 (迁移学习)
    - **大语言模型**: Ollama / Transformers (Qwen2.5)
    - **Web 框架**: Streamlit
    - **编程语言**: Python 3.10+
    - **开发环境**: Windows / Linux
    """)

    st.subheader("💝 鸣谢")
    st.markdown("""
    感谢以下开源项目和数据集的贡献：
    - [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 提供 37 类宠物品种数据
    - [PyTorch](https://pytorch.org/) — 深度学习框架
    - [Streamlit](https://streamlit.io/) — Web 应用框架
    - [Ollama](https://ollama.com/) — 本地大模型运行时
    - [Qwen](https://github.com/QwenLM/Qwen) — 开源大语言模型
    - [HuggingFace Transformers](https://huggingface.co/) — 模型库与推理引擎
    - [Codex CLI](https://github.com/openai/codex) — AI 编程助手，提供代码协助
    """)

    st.subheader("📁 项目结构")
    st.code("""
PetBreedRecognition/
├── data/               # 数据集目录
├── models/             # CNN模型定义、训练与推理
│   ├── train.py        # 训练脚本
│   └── predict.py      # 推理预测
├── llm/                # 大模型交互模块
│   ├── prompts.py      # 提示词模板
│   └── advisor.py      # 养护顾问
├── weights/            # 模型权重文件
├── app.py              # Streamlit 主程序
├── setup.py            # 跨平台一键安装部署脚本
├── requirements.txt    # 依赖库清单
├── run.bat             # Windows 一键运行脚本
└── README.md           # 项目说明文档
    """, language="text")
