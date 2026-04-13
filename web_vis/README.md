# 芝加哥暴力犯罪预测可视化系统

基于ACR-ST模型的Web可视化界面，支持展示预测结果、风险热力图、热点排行和政策建议。

## 功能特性

### 1. 地图可视化
- 网格级别的风险热力图展示
- 四种风险等级颜色编码（极高/高/中/低）
- 支持缩放、平移、点击查看详情
- 悬停高亮网格

### 2. 时间序列
- 支持选择不同日期查看预测结果
- 播放按钮自动轮播多日预测
- 支持未来7天预测展示

### 3. 统计分析
- 当日期望犯罪总数
- 风险分布统计（饼图）
- Top 10%高风险区域数量

### 4. 热点排行
- 按风险分数排序的热点列表
- 点击可飞转到地图位置
- 显示网格ID和风险分数

### 5. 政策建议
- 基于当前风险分布生成建议
- 分级警力配置建议
- 资源配置优先级

## 快速启动

### Windows
双击运行 `start_server.bat`

### 手动启动

```bash
# 1. 进入目录
cd web_vis

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. 安装依赖（简化版，不依赖geopandas）
pip install flask flask-cors numpy

# 4. 准备数据（首次运行）
# 方式A: 完整版（需要geopandas和shapefile）
python prepare_data.py

# 方式B: 简化版（纯Python，无额外依赖）
python prepare_data_simple.py

# 5. 启动服务器
python app.py
```

**注意**: 如果`prepare_data.py`报错（通常是geopandas/fiona兼容性问题），请直接使用`prepare_data_simple.py`，它会生成同样格式的示例数据。

然后打开浏览器访问: **http://localhost:5000**

## 使用说明

### 基本操作
1. **选择日期**: 从下拉菜单选择要查看的预测日期
2. **查看地图**: 网格颜色表示风险等级，点击可查看详情
3. **热点排行**: 右侧列表显示Top 20高风险区域，点击可定位
4. **播放动画**: 点击"播放时间序列"查看多日变化

### 风险等级说明
- 🔴 **极高风险** (E[Y] > 2.0): 建议2辆巡逻车+步行巡逻，每2小时一次
- 🟠 **高风险** (1.0 - 2.0): 建议1辆巡逻车，每4小时一次
- 🟡 **中风险** (0.5 - 1.0): 随机巡逻覆盖
- 🔵 **低风险** (< 0.5): 社区自防

### 实战指标解读
- **Hit Rate 53.4%**: 巡逻Top 10%区域可拦截过半暴力犯罪
- **PAI 3.61**: 单位面积效率是随机巡逻的3.6倍

## 项目结构

```
web_vis/
├── app.py                 # Flask后端API
├── prepare_data.py        # 数据准备脚本
├── requirements.txt       # Python依赖
├── start_server.bat      # Windows启动脚本
├── README.md             # 本文件
├── static/
│   ├── css/             # 样式文件
│   ├── js/              # JavaScript文件
│   └── data/            # GeoJSON和预测数据
└── templates/
    └── index.html       # 主页面
```

## 使用真实模型预测（训练一次，每日推理）

### 方式1: 使用训练好的模型直接推理（推荐）

**训练只需一次**，之后可以反复使用保存的模型进行推理：

```bash
# 1. 进入web_vis目录
cd web_vis

# 2. 生成预测（会自动加载checkpoints/best_model_trans.pt）
python generate_predictions.py

# 3. 启动Web服务器查看结果
python app.py
```

`generate_predictions.py` 会：
- 自动加载训练好的模型权重
- 使用最新数据进行前向传播
- 生成未来7天的预测
- 保存为web可视化需要的格式

### 方式2: 从训练脚本中导出

如果你刚完成训练，可以在 `train_stgcn_trans.py` 末尾添加：

```python
# 保存预测结果供Web可视化使用
predictions = {}
for i in range(7):  # 未来7天
    idx = len(X) - 7 + i
    date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")

    # 获取预测
    pred = pred_test[i] if i < len(pred_test) else np.zeros(num_nodes)

    predictions[date] = {
        "risk_scores": pred.tolist(),
        "risk_levels": [get_risk_level(s) for s in pred],
        "top_10_percent": np.argsort(pred)[-125:].tolist()
    }

# 保存
with open("web_vis/static/data/predictions.json", "w") as f:
    json.dump(predictions, f)
```

### 方式3: 使用独立的推理脚本

更灵活的控制，支持批量预测：

```bash
cd src
python predict.py
```

这会生成7天的预测并保存到 `web_vis/static/data/predictions.json`。

## 每日自动更新流程

建议设置定时任务（crontab或Windows任务计划程序）：

```bash
# 每天凌晨2点自动更新预测
0 2 * * * cd /path/to/web_vis && python generate_predictions.py
```

**工作流程**：
1. 一次性训练模型 → 保存为 `checkpoints/best_model_trans.pt`
2. 每日凌晨自动运行推理脚本 → 更新 `predictions.json`
3. Web界面自动读取最新预测 → 展示在地图上

**无需重新训练！模型权重保持不变，只是前向传播计算预测值。**

## API接口

### GET /api/grid
返回网格GeoJSON数据

### GET /api/predictions
返回可用日期列表

### GET /api/prediction/<date>
返回特定日期的预测详情

**Response:**
```json
{
  "date": "2024-04-10",
  "risk_scores": [0.1, 0.5, 2.3, ...],
  "risk_levels": ["low", "medium", "very_high", ...],
  "top_10_percent": [586, 234, 891, ...],
  "statistics": {
    "very_high": 45,
    "high": 120,
    "medium": 380,
    "low": 701
  },
  "total_expected": 156.8
}
```

### GET /api/prediction/<date>/hotspots
返回热点区域列表

## 技术栈

- **后端**: Flask (Python)
- **前端**: Leaflet.js (地图), Bootstrap 5 (UI)
- **地图**: OpenStreetMap底图
- **数据格式**: GeoJSON

## 性能说明

- 支持1246个网格流畅渲染
- 初始加载时间约2-3秒
- 日期切换响应时间 < 500ms
- 支持7天时间序列动画播放

## 浏览器兼容性

- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

## 后续优化方向

1. **实时数据接入**: 对接实时犯罪报警系统
2. **近重复响应**: 枪击事件后自动高亮周边区域
3. **移动端适配**: 优化手机浏览器体验
4. **导出功能**: 支持导出巡逻路线PDF
5. **用户认证**: 添加登录系统区分权限

## 截图示例

### 主界面
- 左侧：日期选择、统计信息、图例
- 中间：交互式风险热力图
- 右侧：热点排行、政策建议

### 典型使用流程
1. 选择预测日期
2. 查看地图高亮区域
3. 点击热点查看详情
4. 根据政策建议调整巡逻计划

## 联系与支持

如有问题，请查看项目README或提交Issue。
