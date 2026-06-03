@echo off
chcp 65001 > nul
echo ========================================
echo   宠物品种识别与养护建议系统 - 一键启动
echo ========================================
echo.

:: 检查 Python 环境
python --version > nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python 环境！
    echo 请先安装 Python 3.10+ 并添加到系统 PATH
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] 检查 Python 环境...
python --version

echo.
echo [2/3] 安装项目依赖...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，尝试继续运行...
)

echo.
echo [3/3] 启动应用...
echo 请在浏览器中打开显示的地址（默认 http://localhost:8501）
echo 按 Ctrl+C 停止运行
echo.
streamlit run app.py --server.port 8501

pause
