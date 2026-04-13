@echo off
echo ========================================
echo 芝加哥暴力犯罪预测可视化系统
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查虚拟环境
if not exist "venv" (
    echo [1/4] 创建虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
echo [2/4] 激活虚拟环境...
call venv\Scripts\activate.bat

REM 安装依赖
echo [3/4] 安装依赖...
pip install -q flask flask-cors numpy

REM 安装可选依赖（geopandas可能安装失败但不影响运行）
pip install -q geopandas 2>nul || echo [提示] geopandas安装失败，将使用简化数据生成

REM 准备数据
echo [4/4] 准备数据...
if not exist "static\data\grid_metadata.json" (
    echo 尝试使用完整数据准备...
    python prepare_data.py
    if errorlevel 1 (
        echo [警告] 完整数据准备失败，使用简化版本...
        python prepare_data_simple.py
    )
) else (
    echo 数据已存在，跳过准备步骤
)

echo.
echo ========================================
echo 启动Web服务器...
echo 请在浏览器中访问: http://localhost:5000
echo ========================================
python app.py

pause
