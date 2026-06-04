# 智能宠物品种识别与养护建议系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)

基于 **ResNet-18 迁移学习** 与 **大语言模型** 的宠物品种识别系统，上传照片即可自动识别 37 种猫狗品种，并生成个性化养护建议。

## 目录

- [功能特性](#功能特性)
- [可识别品种](#可识别品种)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [大模型配置](#大模型配置)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [贡献者](#贡献者)
- [鸣谢](#鸣谢)

## 功能特性

| 功能 | 描述 |
|------|------|
| 📸 品种识别 | 上传宠物照片，CNN 自动识别品种，展示 Top-5 及置信度 |
| 💬 养护顾问 | 大模型生成饮食、护理、健康、训练全方位建议 |
| 📊 品种对比 | 两品种多维对比（体型、性格、饲养难度等） |
| 🏥 健康咨询 | 常见健康问题分析与应急参考建议 |

## 可识别品种

共 **37 种**，基于 [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) 训练。

### 猫（12 种）

| Abyssinian | 阿比西尼亚猫 | Bengal | 孟加拉猫 |
| Birman | 伯曼猫 | Bombay | 孟买猫 |
| British Shorthair | 英国短毛猫 | Egyptian Mau | 埃及猫 |
| Maine Coon | 缅因猫 | Persian | 波斯猫 |
| Ragdoll | 布偶猫 | Russian Blue | 俄罗斯蓝猫 |
| Siamese | 暹罗猫 | Sphynx | 斯芬克斯猫 |

### 狗（25 种）

| American Bulldog | 美国斗牛犬 | American Pit Bull Terrier | 美国比特犬 |
| Basset Hound | 巴吉度猎犬 | Beagle | 比格犬 |
| Boxer | 拳师犬 | Chihuahua | 吉娃娃 |
| English Cocker Spaniel | 英国可卡犬 | English Setter | 英国雪达犬 |
| German Shorthaired | 德国短毛指示犬 | Great Pyrenees | 大白熊犬 |
| Havanese | 哈瓦那犬 | Japanese Chin | 日本狆 |
| Keeshond | 荷兰毛狮犬 | Leonberger | 莱昂贝格犬 |
| Miniature Pinscher | 迷你品犬 | Newfoundland | 纽芬兰犬 |
| Pomeranian | 博美犬 | Pug | 巴哥犬 |
| Saint Bernard | 圣伯纳犬 | Samoyed | 萨摩耶犬 |
| Scottish Terrier | 苏格兰梗 | Shiba Inu | 柴犬 |
| Staffordshire Bull Terrier | 斯塔福郡斗牛梗 | Wheaten Terrier | 爱尔兰软毛梗 |
| Yorkshire Terrier | 约克夏梗 | | |

## 快速开始

### 环境要求

- **Python** ≥ 3.10（[下载](https://www.python.org/downloads/)，安装时勾选 `Add Python to PATH`）

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourname/PetBreedRecognition.git
cd PetBreedRecognition

# 安装 PyTorch（CPU 版约 200MB；有 NVIDIA 显卡将 /cpu 去掉）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其余依赖
pip install -r requirements.txt
```

> 国内用户可加清华镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 启动

```bash
streamlit run app.py
```

浏览器访问 **http://localhost:8501**。Windows 用户也可直接双击 `run.bat`。

## 使用说明

1. 左侧边栏选择功能模块
2. **品种识别** — 上传 JPG/PNG 照片，点击识别，柱状图展示 Top-5 预测
3. **养护顾问** — 识别后自动带入品种，选择话题方向即可对话
4. **品种对比** — 输入两个品种名，一键生成多维对比
5. **健康咨询** — 输入品种和症状，获取参考建议

> ⚠️ AI 建议仅供参考，不能替代专业兽医诊断。

## 大模型配置

系统支持两种 LLM 后端，在侧边栏切换：

### Ollama（需要大模型能力时推荐）

```bash
# 安装: https://ollama.com/
ollama pull qwen2.5:1.5b
```

侧边栏选择 `ollama`，模型名填 `qwen2.5:1.5b`。

### Fallback（无需任何配置）

选择 `fallback` 使用离线规则库，不依赖网络和外部服务。

## 项目结构

```
PetBreedRecognition/
├── models/
│   ├── train.py          # 模型训练
│   └── predict.py        # 推理与品种映射
├── llm/
│   ├── prompts.py        # 提示词模板
│   └── advisor.py        # 养护顾问后端
├── weights/              # 模型权重文件 (*.pth)
├── app.py                # Streamlit 主入口
├── requirements.txt      # Python 依赖
├── run.bat               # Windows 快速启动
└── README.md
```

## 常见问题

<details>
<summary><b>启动报错 No module named 'xxx'</b></summary>
依赖未完整安装，重新执行安装步骤。
</details>

<details>
<summary><b>预测图表中文显示为方块</b></summary>
已配置微软雅黑/黑体/楷体字体。如仍异常，检查系统是否安装了上述字体之一。
</details>

<details>
<summary><b>大模型功能无响应</b></summary>
侧边栏切换到 `fallback` 模式即可离线使用。
</details>

<details>
<summary><b>端口 8501 被占用</b></summary>

```bash
streamlit run app.py --server.port 8502
```
</details>

## 贡献者

| 贡献者 | 职责 |
|--------|------|
| [小小诺亚](https://github.com/) | 系统架构、模型训练、前端开发 |
| [Legend2333](https://github.com/) | 大模型集成、提示词工程、测试 |

## 鸣谢

本项目得益于以下开源项目：

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 训练数据
- [PyTorch](https://pytorch.org/) — 深度学习框架
- [Streamlit](https://streamlit.io/) — Web 应用框架
- [Ollama](https://ollama.com/) / [Qwen](https://github.com/QwenLM/Qwen) — 本地大模型
- [Codex CLI](https://github.com/openai/codex) — AI 编程助手

## 许可证

[MIT](LICENSE)