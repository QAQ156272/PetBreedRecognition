# 🐾 智能宠物品种识别与养护建议系统

## 📋 项目简介

基于深度学习和自然语言处理技术，构建智能宠物品种识别与养护建议系统。用户上传宠物照片后，系统基于 ResNet-18 模型自动识别品种，并结合大语言模型生成个性化的饮食、护理、健康、训练等全方位养护建议。

## 🎯 可识别品种（37 种）

### 🐱 猫咪品种（12 种）

| 英文名 | 中文名 | 英文名 | 中文名 |
|--------|--------|--------|--------|
| Abyssinian | 阿比西尼亚猫 | Bengal | 孟加拉猫 |
| Birman | 伯曼猫 | Bombay | 孟买猫 |
| British Shorthair | 英国短毛猫 | Egyptian Mau | 埃及猫 |
| Maine Coon | 缅因猫 | Persian | 波斯猫 |
| Ragdoll | 布偶猫 | Russian Blue | 俄罗斯蓝猫 |
| Siamese | 暹罗猫 | Sphynx | 斯芬克斯猫 |

### 🐶 狗狗品种（25 种）

| 英文名 | 中文名 | 英文名 | 中文名 |
|--------|--------|--------|--------|
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

## 🚀 操作指南

### 环境要求

| 组件 | 版本要求 | 说明 |
|------|----------|------|
| Python | 3.10 - 3.11 | 必须，[下载地址](https://www.python.org/downloads/) |
| 操作系统 | Windows 10/11 或 Linux | — |
| Ollama | 最新版 | 可选，用于本地大模型生成养护建议 |

### 方式一：一键运行（Windows 推荐）

1. 确保已安装 Python 3.10 - 3.11，并勾选了 **"Add Python to PATH"**
2. 双击 `run.bat`
3. 脚本会自动完成：
   - 检测 Python 环境
   - 安装项目依赖（使用清华镜像加速）
   - 启动 Streamlit 应用
4. 浏览器访问 **http://localhost:8501**

### 方式二：手动运行

```bash
# 1. 克隆或下载项目到本地

# 2. 进入项目目录
cd PetBreedRecognition

# 3. 安装依赖（推荐使用清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 启动应用
streamlit run app.py

# 5. 浏览器访问 http://localhost:8501
```

### 使用流程

1. **侧边栏选择功能页面** — 首页 / 品种识别 / 养护顾问 / 品种对比 / 健康咨询
2. **品种识别**：上传宠物照片（JPG / PNG），点击"开始识别"，查看 Top-5 预测结果和置信度
3. **养护顾问**：识别后自动跳转，选择咨询方向（饮食/护理/健康/训练），获取大模型生成的建议
4. **品种对比**：输入两个品种名称，大模型从多个维度进行对比分析
5. **健康咨询**：输入品种和疑问，获取健康相关的参考建议

> ⚠️ **注意**：养护顾问和健康咨询等功能需要配置大模型后端（详见下方说明）。

## ⚙️ 大模型配置

系统在侧边栏提供三种 LLM 后端模式：

### 1. Ollama（推荐，本地轻量）

```bash
# 安装 Ollama
# 官网下载：https://ollama.com/

# 拉取模型
ollama pull qwen2.5:1.5b

# 启动服务（通常自动运行）
ollama serve
```

然后在侧边栏选择 **ollama**，模型名默认 `qwen2.5:1.5b`。

### 2. Transformers（HuggingFace 直接加载）

1. 取消 `requirements.txt` 中 transformers 相关三行的注释：
   ```
   transformers==4.38.1
   accelerate==0.28.0
   sentencepiece==0.1.99
   ```
2. 重新 `pip install -r requirements.txt`
3. 侧边栏选择 **transformers**，模型名默认 `Qwen/Qwen2.5-0.5B-Instruct`

### 3. Fallback（离线规则模式）

无需额外配置，选择 **fallback** 即可使用内置规则库。不依赖任何外部服务，但建议内容较为通用。

## 🏗️ 项目结构

```
PetBreedRecognition/
├── data/               # 数据集目录（Oxford-IIIT Pet Dataset）
├── models/             # CNN 模型模块
│   ├── train.py        # 模型训练脚本
│   └── predict.py      # 推理预测模块
├── llm/                # 大模型交互模块
│   ├── prompts.py      # 提示词模板
│   └── advisor.py      # 养护顾问
├── weights/            # 模型权重文件（.pth）
├── app.py              # Streamlit 主程序
├── requirements.txt    # Python 依赖清单
├── run.bat             # Windows 一键运行脚本
└── README.md           # 本文件
```

## 🔧 功能模块

| 功能 | 说明 |
|------|------|
| 📸 品种识别 | 上传宠物照片，CNN 模型自动识别品种，展示 Top-5 预测及置信度 |
| 💬 养护顾问 | 大模型生成饮食、护理、健康、训练全方位个性化建议 |
| 📊 品种对比 | 对比不同品种特点，从体型、性格、饲养难度等维度分析 |
| 🏥 健康咨询 | 提供常见健康问题分析与应急处理参考建议 |

## 📊 技术指标

| 指标 | 数值 |
|------|------|
| 可识别品种 | 37 种（猫 12 种 + 狗 25 种） |
| 模型架构 | ResNet-18（迁移学习） |
| 目标准确率 | ≥ 92% |
| 单图推理时间 | < 1 秒 |
| 数据集 | Oxford-IIIT Pet Dataset |

## 👥 贡献者

| 角色 | 职责 |
|------|------|
| 🐱 小小诺亚 | 项目架构设计、模型开发与训练、前端开发 |
| 🐶 Legend2333 | 大模型集成、提示词工程、文档与测试 |

## ❓ 常见问题

**Q: 启动报错 `No module named 'xxx'`？**
A: 依赖未完整安装，执行 `pip install -r requirements.txt` 重新安装。

**Q: 中文显示为方块？**
A: 已在 `app.py` 中配置了 matplotlib 中文字体（微软雅黑/黑体/楷体），确保系统安装了其中至少一种。

**Q: 大模型功能无法使用？**
A: 检查侧边栏 LLM 配置是否匹配当前环境。如未安装 Ollama，切换到 **fallback** 模式。

**Q: 端口 8501 被占用？**
A: 运行时加参数指定其他端口：`streamlit run app.py --server.port 8502`


## 💝 鸣谢

感谢以下开源项目和数据集的贡献：

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) — 提供 37 类宠物品种数据
- [PyTorch](https://pytorch.org/) — 深度学习框架
- [Streamlit](https://streamlit.io/) — Web 应用框架
- [Ollama](https://ollama.com/) — 本地大模型运行时
- [Qwen](https://github.com/QwenLM/Qwen) — 开源大语言模型
- [HuggingFace Transformers](https://huggingface.co/) — 模型库与推理引擎
- [Codex CLI](https://github.com/openai/codex) — AI 编程助手，提供代码协助
## 📝 许可证

本项目采用 MIT 许可证，数据集版权归原作者所有。