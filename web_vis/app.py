"""
Web可视化服务器
- 提供预测数据API
- 支持查看历史和未来预测
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
CORS(app)

# 加载数据
GRID_GEOJSON = None
PREDICTIONS = None
GRID_METADATA = None


def load_data():
    """加载所有必要数据"""
    global GRID_GEOJSON, PREDICTIONS, GRID_METADATA

    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载网格GeoJSON
    grid_path = os.path.join(script_dir, "static", "data", "chicago_grid.geojson")
    if os.path.exists(grid_path):
        try:
            with open(grid_path, 'r', encoding='utf-8') as f:
                GRID_GEOJSON = json.load(f)
            print(f"Loaded grid data: {len(GRID_GEOJSON['features'])} cells")
        except Exception as e:
            print(f"Error loading grid GeoJSON: {e}")
            # 尝试生成简化版GeoJSON从metadata
            GRID_GEOJSON = generate_geojson_from_metadata()
    else:
        print("Warning: Grid GeoJSON not found. Trying to generate from metadata...")
        GRID_GEOJSON = generate_geojson_from_metadata()

    # 加载预测数据
    pred_path = os.path.join(script_dir, "static", "data", "predictions.json")
    if os.path.exists(pred_path):
        try:
            with open(pred_path, 'r', encoding='utf-8') as f:
                PREDICTIONS = json.load(f)
            print(f"Loaded predictions: {len(PREDICTIONS)} days")
        except Exception as e:
            print(f"Error loading predictions: {e}")
    else:
        print("Warning: Predictions not found. Run prepare_data.py first.")

    # 加载网格元数据
    meta_path = os.path.join(script_dir, "static", "data", "grid_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                GRID_METADATA = json.load(f)
            print(f"Loaded metadata: {len(GRID_METADATA)} cells")
        except Exception as e:
            print(f"Error loading metadata: {e}")


def generate_geojson_from_metadata():
    """从metadata生成简化版GeoJSON"""
    global GRID_METADATA

    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(script_dir, "static", "data", "grid_metadata.json")

    if not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            GRID_METADATA = json.load(f)

        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        for meta in GRID_METADATA:
            lat = meta["center_lat"]
            lon = meta["center_lon"]
            grid_id = meta["grid_id"]

            # 创建简化的正方形网格（约500m）
            delta = 0.0025  # 约250m的一半
            feature = {
                "type": "Feature",
                "properties": {"grid_id": grid_id},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon - delta, lat - delta],
                        [lon + delta, lat - delta],
                        [lon + delta, lat + delta],
                        [lon - delta, lat + delta],
                        [lon - delta, lat - delta]
                    ]]
                }
            }
            geojson["features"].append(feature)

        print(f"Generated GeoJSON from metadata: {len(geojson['features'])} cells")
        return geojson

    except Exception as e:
        print(f"Error generating GeoJSON from metadata: {e}")
        return None


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/tech')
def index_tech():
    """科技风格主页面"""
    return render_template('index_tech.html')


@app.route('/api/grid')
def get_grid():
    """获取网格GeoJSON数据"""
    if GRID_GEOJSON is None:
        return jsonify({"error": "Grid data not loaded"}), 500
    return jsonify(GRID_GEOJSON)


@app.route('/api/predictions')
def get_predictions():
    """获取所有预测日期列表"""
    if PREDICTIONS is None:
        return jsonify({"error": "Predictions not loaded"}), 500

    dates = list(PREDICTIONS.keys())
    dates.sort()

    return jsonify({
        "dates": dates,
        "total_grids": len(GRID_METADATA) if GRID_METADATA else 0
    })


@app.route('/api/prediction/<date>')
def get_prediction_by_date(date):
    """获取特定日期的预测结果"""
    if PREDICTIONS is None:
        return jsonify({"error": "Predictions not loaded"}), 500

    if date not in PREDICTIONS:
        return jsonify({"error": f"No prediction for date {date}"}), 404

    pred = PREDICTIONS[date]

    # 统计信息
    risk_counts = {
        "very_high": pred["risk_levels"].count("very_high"),
        "high": pred["risk_levels"].count("high"),
        "medium": pred["risk_levels"].count("medium"),
        "low": pred["risk_levels"].count("low")
    }

    return jsonify({
        "date": date,
        "risk_scores": pred["risk_scores"],
        "risk_levels": pred["risk_levels"],
        "top_10_percent": pred["top_10_percent"],
        "statistics": risk_counts,
        "total_expected": sum(pred["risk_scores"])
    })


@app.route('/api/prediction/<date>/hotspots')
def get_hotspots(date):
    """获取特定日期的高风险热点区域"""
    if PREDICTIONS is None or GRID_METADATA is None:
        return jsonify({"error": "Data not loaded"}), 500

    if date not in PREDICTIONS:
        return jsonify({"error": f"No prediction for date {date}"}), 404

    pred = PREDICTIONS[date]
    hotspots = []

    for grid_id in pred["top_10_percent"]:
        if grid_id < len(GRID_METADATA):
            meta = GRID_METADATA[grid_id]
            hotspots.append({
                "grid_id": grid_id,
                "lat": meta["center_lat"],
                "lon": meta["center_lon"],
                "risk_score": pred["risk_scores"][grid_id],
                "risk_level": pred["risk_levels"][grid_id]
            })

    # 按风险分数排序
    hotspots.sort(key=lambda x: x["risk_score"], reverse=True)

    return jsonify({
        "date": date,
        "hotspots": hotspots,
        "count": len(hotspots)
    })


@app.route('/api/summary')
def get_summary():
    """获取预测摘要统计"""
    if PREDICTIONS is None:
        return jsonify({"error": "Predictions not loaded"}), 500

    dates = sorted(PREDICTIONS.keys())

    # 计算总体统计
    total_crimes = []
    for date in dates:
        total_crimes.append(sum(PREDICTIONS[date]["risk_scores"]))

    return jsonify({
        "date_range": f"{dates[0]} to {dates[-1]}",
        "total_days": len(dates),
        "avg_daily_crimes": round(np.mean(total_crimes), 2),
        "max_daily_crimes": round(max(total_crimes), 2),
        "min_daily_crimes": round(min(total_crimes), 2)
    })


@app.route('/api/grid/<int:grid_id>/recommendations')
def get_grid_recommendations(grid_id):
    """
    获取指定网格的政策建议
    基于网格特征生成具体的警力部署和环境改善建议
    """
    if PREDICTIONS is None or GRID_METADATA is None:
        return jsonify({"error": "Data not loaded"}), 500

    if grid_id < 0 or grid_id >= len(GRID_METADATA):
        return jsonify({"error": f"Invalid grid_id {grid_id}"}), 404

    # 获取网格元数据
    meta = GRID_METADATA[grid_id]

    # 获取最新日期的预测风险
    latest_date = sorted(PREDICTIONS.keys())[-1]
    risk_score = PREDICTIONS[latest_date]["risk_scores"][grid_id]
    risk_level = PREDICTIONS[latest_date]["risk_levels"][grid_id]

    # 基于风险等级和网格特征生成建议
    # 简化版：直接基于风险等级返回预设建议
    recommendations = generate_simple_recommendations(
        grid_id, risk_score, risk_level, meta
    )

    return jsonify(recommendations)


def generate_simple_recommendations(grid_id, risk_score, risk_level, meta):
    """
    生成简化的政策建议（无需依赖完整特征数据）
    基于网格风险等级和基本元数据
    """
    # 根据风险等级确定建议
    if risk_level == "very_high":
        immediate_actions = [
            {
                "action": "启动近重复预警响应",
                "detail": "在该网格及周边500米实施超常规巡逻（每2小时一次）",
                "priority": "极高",
                "cost": "低"
            },
            {
                "action": "部署定点警力",
                "detail": f"在网格中心({meta.get('center_lat', 0):.4f}, {meta.get('center_lon', 0):.4f})设置警车定点停靠",
                "priority": "高",
                "cost": "低"
            },
            {
                "action": "增派便衣巡逻",
                "detail": "部署2-3名便衣警员在重点区域（商业区出入口、公交站点）",
                "priority": "高",
                "cost": "低"
            }
        ]
        short_term = [
            "检查并增补监控摄像头覆盖",
            "联系商户建立联防机制",
            "评估照明状况并升级路灯"
        ]
        diagnosis = "极高风险区域，建议立即启动一级响应"

    elif risk_level == "high":
        immediate_actions = [
            {
                "action": "增加巡逻频次",
                "detail": "将该网格纳入重点巡逻路线，每4小时巡逻一次",
                "priority": "高",
                "cost": "低"
            },
            {
                "action": "设置临时监控",
                "detail": "在该区域临时部署移动监控设备",
                "priority": "中",
                "cost": "中"
            }
        ]
        short_term = [
            "组织社区安全会议",
            "检查建筑门禁系统",
            "修剪遮挡视线的植被"
        ]
        diagnosis = "高风险区域，需要加强警力和社区合作"

    elif risk_level == "medium":
        immediate_actions = [
            {
                "action": "常规巡逻关注",
                "detail": "保持正常巡逻频次，但重点关注该区域异常活动",
                "priority": "中",
                "cost": "低"
            }
        ]
        short_term = [
            "开展社区防范宣传",
            "检查公共设施完好性"
        ]
        diagnosis = "中等风险，建议保持关注"

    else:
        immediate_actions = []
        short_term = ["维持现有防控措施"]
        diagnosis = "低风险区域"

    return {
        "grid_id": grid_id,
        "location": {
            "lat": meta.get("center_lat", 0),
            "lon": meta.get("center_lon", 0)
        },
        "risk_assessment": {
            "level": risk_level,
            "score": round(risk_score, 4),
            "diagnosis": diagnosis
        },
        "immediate_actions": immediate_actions,
        "short_term_actions": short_term,
        "environmental_suggestions": get_environmental_suggestions(risk_level),
        "estimated_budget": {
            "manpower": "高" if risk_level == "very_high" else ("中" if risk_level == "high" else "低"),
            "infrastructure": "视具体措施而定"
        }
    }


def get_environmental_suggestions(risk_level):
    """根据风险等级返回环境改善建议"""
    suggestions = {
        "lighting": {
            "problem": "照明不足可能提供犯罪机会",
            "solutions": [
                "升级LED路灯至200W以上",
                "消除照明死角（修剪树枝、清洁灯罩）",
                "鼓励商户夜间亮灯"
            ]
        },
        "surveillance": {
            "problem": "监护盲区可能导致犯罪不易被发现",
            "solutions": [
                "安装AI智能摄像头覆盖盲区",
                "组织社区守望计划",
                "增加巡警可见度"
            ]
        },
        "greening": {
            "problem": "绿化设计不当可能形成藏匿点",
            "solutions": [
                "修剪灌木至0.6米以下",
                "开辟视线通透廊道（确保15米可见）",
                "在空旷区域增设口袋公园增加正当人流"
            ]
        },
        "access_control": {
            "problem": "入口控制不足便于逃逸",
            "solutions": [
                "封闭不必要的后巷",
                "在关键路径设置自然障碍物",
                "优化道路设计减少逃逸通道"
            ]
        }
    }

    # 根据风险等级返回重点建议
    if risk_level == "very_high":
        return {
            "priority": "立即行动",
            "focus_areas": ["lighting", "surveillance", "access_control"],
            "detailed_suggestions": suggestions
        }
    elif risk_level == "high":
        return {
            "priority": "短期改进",
            "focus_areas": ["surveillance", "greening"],
            "detailed_suggestions": {k: suggestions[k] for k in ["surveillance", "greening"]}
        }
    else:
        return {
            "priority": "维持现状",
            "focus_areas": [],
            "detailed_suggestions": {}
        }


if __name__ == '__main__':
    print("="*60)
    print("Crime Prediction Web Visualization Server")
    print("="*60)

    load_data()

    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5000)
