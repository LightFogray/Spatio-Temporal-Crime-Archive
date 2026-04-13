#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芝加哥犯罪数据下载脚本 (2021-2024)
=====================================

功能：
1. 按犯罪类型分类下载数据（暴力犯罪、财产犯罪）
2. 分块下载避免API限流
3. 自动数据清洗和验证
4. 生成数据质量报告
5. 支持断点续传

作者：犯罪地理学数据分析系统
版本：2.0
日期：2026-04-12
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sodapy import Socrata
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==================== 配置区域 ====================

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('chicago_crime_download.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 芝加哥犯罪数据集配置
DATASET_ID = "ijzp-q8t2"
BASE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2"

# 犯罪类型分类
CRIME_CATEGORIES = {
    # 暴力犯罪 (Violent Crime)
    "violent": {
        "name": "暴力犯罪",
        "description": "涉及对受害者使用武力或武力威胁的犯罪",
        "primary_types": [
            'HOMICIDE',                    # 凶杀
            'CRIM SEXUAL ASSAULT',         # 刑事性侵
            'CRIMINAL SEXUAL ASSAULT',     # 刑事性侵（备用名称）
            'ROBBERY',                     # 抢劫
            'ASSAULT',                     # 袭击/攻击
            'BATTERY',                     # 殴打/人身伤害
            'KIDNAPPING',                  # 绑架
            'INTIMIDATION',                # 恐吓
            'STALKING',                    # 跟踪
            'HUMAN TRAFFICKING'            # 人口贩卖
        ]
    },

    # 财产犯罪 (Property Crime)
    "property": {
        "name": "财产犯罪",
        "description": "涉及财物的非法取得或破坏的犯罪",
        "primary_types": [
            'THEFT',                       # 盗窃
            'BURGLARY',                    # 入室盗窃
            'MOTOR VEHICLE THEFT',         # 机动车盗窃
            'ARSON',                       # 纵火
            'DECEPTIVE PRACTICE',          # 欺诈/诈骗
            'CRIMINAL DAMAGE',             # 刑事毁坏/破坏公物
            'CRIMINAL TRESPASS'            # 非法侵入
        ]
    }
}

# 时间范围
START_YEAR = 2022
END_YEAR = 2023

# API配置
API_LIMIT_PER_REQUEST = 50000  # 单次请求最大记录数
REQUEST_DELAY = 2.0  # 请求间隔（秒）
MAX_RETRIES = 5  # 最大重试次数
TIMEOUT = 60  # 超时时间（秒）

# 输出目录
OUTPUT_DIR = "chicago_crime_data"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
CLEANED_DIR = os.path.join(OUTPUT_DIR, "cleaned")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# 创建目录
for directory in [OUTPUT_DIR, RAW_DIR, CLEANED_DIR, REPORTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)


# ==================== 工具函数 ====================

def setup_session() -> requests.Session:
    """创建带重试机制的HTTP会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_primary_types_query(primary_types: List[str]) -> str:
    """生成primary_type的WHERE子句"""
    escaped_types = [f"'{pt}'" for pt in primary_types]
    return f"primary_type in ({','.join(escaped_types)})"


def get_year_query(year: int) -> str:
    """生成年份的WHERE子句"""
    return f"year = {year}"


def build_where_clause(primary_types: List[str], year: int) -> str:
    """构建完整的WHERE子句"""
    type_clause = get_primary_types_query(primary_types)
    year_clause = get_year_query(year)
    return f"({type_clause}) AND ({year_clause})"


def calculate_chunks(total_records: int) -> List[Tuple[int, int]]:
    """
    计算分块下载的偏移量

    Args:
        total_records: 总记录数

    Returns:
        List of (offset, limit) tuples
    """
    chunks = []
    offset = 0

    while offset < total_records:
        limit = min(API_LIMIT_PER_REQUEST, total_records - offset)
        chunks.append((offset, limit))
        offset += API_LIMIT_PER_REQUEST

    return chunks


def save_checkpoint(category: str, year: int, offset: int, records: List[Dict]):
    """保存断点数据到临时文件"""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{category}_{year}_checkpoint_{offset}.json")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)
    logger.debug(f"保存断点: {checkpoint_file} ({len(records)} 条记录)")


def load_checkpoints(category: str, year: int) -> List[Dict]:
    """加载所有断点数据"""
    all_records = []
    pattern = f"{category}_{year}_checkpoint_"

    checkpoint_files = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith(pattern) and f.endswith('.json')
    ]

    checkpoint_files.sort()

    for filename in checkpoint_files:
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                records = json.load(f)
                all_records.extend(records)
                logger.debug(f"加载断点: {filename} ({len(records)} 条记录)")
        except Exception as e:
            logger.warning(f"加载断点文件失败 {filename}: {e}")

    return all_records


def clear_checkpoints(category: str, year: int):
    """清除断点文件"""
    pattern = f"{category}_{year}_checkpoint_"
    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.startswith(pattern) and filename.endswith('.json'):
            try:
                os.remove(os.path.join(CHECKPOINT_DIR, filename))
            except Exception:
                pass


# ==================== 数据下载类 ====================

class ChicagoCrimeDownloader:
    """芝加哥犯罪数据下载器"""

    def __init__(self):
        self.client = Socrata("data.cityofchicago.org", None)
        self.session = setup_session()
        self.download_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_records": 0,
            "start_time": None,
            "end_time": None
        }

    def get_actual_record_count(self, category: str, year: int) -> int:
        """
        获取实际的记录总数

        Args:
            category: 犯罪类别
            year: 年份

        Returns:
            实际记录总数
        """
        primary_types = CRIME_CATEGORIES[category]["primary_types"]
        where_clause = build_where_clause(primary_types, year)

        try:
            # 使用count查询获取实际记录数
            params = {
                "$select": "count(*)",
                "$where": where_clause
            }
            result = self.client.get(DATASET_ID, **params)

            if result and len(result) > 0:
                count = int(result[0].get('count', 0))
                logger.info(f"  实际记录数查询结果: {count:,} 条")
                return count
            else:
                logger.warning("  查询记录数返回空结果")
                return 0

        except Exception as e:
            logger.error(f"  查询记录数失败: {e}")
            return 0

    def download_year_category(
        self,
        category: str,
        year: int,
        force_redownload: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        下载指定类别和年份的犯罪数据

        Args:
            category: 犯罪类别 ('violent', 'property')
            year: 年份
            force_redownload: 是否强制重新下载

        Returns:
            DataFrame containing the crime data, or None if failed
        """
        # 检查文件是否已存在
        output_file = os.path.join(RAW_DIR, f"{category}_{year}.csv")
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{category}_{year}_progress.json")

        if os.path.exists(output_file) and not force_redownload:
            logger.info(f"[跳过] 文件已存在: {output_file}")
            try:
                df = pd.read_csv(output_file)
                logger.info(f"  已加载 {len(df):,} 条记录")
                return df
            except Exception as e:
                logger.warning(f"  读取文件失败: {e}，重新下载")

        # 清除旧断点（如果强制重新下载）
        if force_redownload:
            clear_checkpoints(category, year)
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)

        # 获取犯罪类型列表
        primary_types = CRIME_CATEGORIES[category]["primary_types"]

        # 获取实际记录数（而非估算）
        logger.info(f"\n{'='*60}")
        logger.info(f"开始下载: {CRIME_CATEGORIES[category]['name']} - {year}年")
        logger.info(f"犯罪类型: {len(primary_types)} 种")
        logger.info(f"{'='*60}")

        actual_records = self.get_actual_record_count(category, year)

        if actual_records == 0:
            logger.warning(f"警告: 未找到 {category} - {year} 年的数据")
            return None

        chunks = calculate_chunks(actual_records)

        logger.info(f"分块数: {len(chunks)}")
        logger.info(f"每块大小: {API_LIMIT_PER_REQUEST:,}")

        # 加载之前的进度
        completed_chunks = set()
        if os.path.exists(checkpoint_file) and not force_redownload:
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    completed_chunks = set(progress.get('completed_chunks', []))
                    logger.info(f"恢复进度: {len(completed_chunks)}/{len(chunks)} 块已下载")
            except Exception as e:
                logger.warning(f"读取进度文件失败: {e}")

        all_records = []
        total_downloaded = 0

        # 构建WHERE子句
        where_clause = build_where_clause(primary_types, year)

        # 需要查询的字段
        select_fields = [
            "id", "case_number", "date", "block", "iucr",
            "primary_type", "description", "location_description",
            "arrest", "domestic", "beat", "district", "ward",
            "community_area", "fbi_code", "x_coordinate", "y_coordinate",
            "year", "updated_on", "latitude", "longitude", "location"
        ]

        # 分块下载
        for i, (offset, limit) in enumerate(chunks, 1):
            # 跳过已完成的块
            if i in completed_chunks:
                logger.info(f"[块 {i}/{len(chunks)}] 已下载，跳过")
                # 从断点加载
                checkpoint_records = load_checkpoints(category, year)
                for record in checkpoint_records:
                    if record.get('_chunk_id') == i:
                        all_records.append(record)
                        total_downloaded += 1
                continue

            max_chunk_retries = 3
            chunk_success = False

            for retry in range(max_chunk_retries):
                try:
                    self.download_stats["total_requests"] += 1

                    logger.info(f"[块 {i}/{len(chunks)}] 偏移量: {offset:,}, 限制: {limit:,}")

                    # 构建查询参数
                    params = {
                        "$select": ",".join(select_fields),
                        "$where": where_clause,
                        "$order": "date DESC",
                        "$limit": limit,
                        "$offset": offset
                    }

                    # 执行查询
                    results = self.client.get(DATASET_ID, **params)

                    if results:
                        # 标记块ID用于断点恢复
                        for record in results:
                            record['_chunk_id'] = i

                        all_records.extend(results)
                        total_downloaded += len(results)
                        self.download_stats["successful_requests"] += 1

                        # 保存断点
                        save_checkpoint(category, year, offset, results)
                        completed_chunks.add(i)

                        # 保存进度
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                'completed_chunks': list(completed_chunks),
                                'total_chunks': len(chunks),
                                'total_downloaded': total_downloaded
                            }, f)

                        logger.info(f"  成功下载 {len(results):,} 条，累计: {total_downloaded:,}")
                        chunk_success = True
                    else:
                        logger.warning(f"  块 {i} 返回空结果")
                        if total_downloaded >= actual_records * 0.95:  # 95%完成度就认为是完成了
                            logger.info("  数据已接近完整，停止下载")
                            chunk_success = True
                            break

                    # 避免API限流
                    if i < len(chunks):
                        time.sleep(REQUEST_DELAY)

                    break  # 成功，跳出重试循环

                except Exception as e:
                    self.download_stats["failed_requests"] += 1
                    logger.error(f"  块 {i} 下载失败 (尝试 {retry+1}/{max_chunk_retries}): {e}")

                    # 如果是API限流，等待更长时间
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        wait_time = 30 * (retry + 1)
                        logger.info(f"    检测到API限流，等待 {wait_time} 秒...")
                        time.sleep(wait_time)
                    elif retry < max_chunk_retries - 1:
                        time.sleep(5 * (retry + 1))

                    if retry == max_chunk_retries - 1:
                        logger.error(f"  块 {i} 下载失败，已达到最大重试次数")

            if not chunk_success and total_downloaded > 0:
                logger.warning(f"  部分数据下载失败，已下载 {total_downloaded:,} 条")

        # 移除临时标记字段
        for record in all_records:
            record.pop('_chunk_id', None)

        # 转换为DataFrame
        if all_records:
            df = pd.DataFrame.from_records(all_records)

            # 去重（基于id字段）
            if 'id' in df.columns:
                before_dedup = len(df)
                df = df.drop_duplicates(subset=['id'])
                after_dedup = len(df)
                if before_dedup != after_dedup:
                    logger.info(f"去重: {before_dedup - after_dedup} 条重复记录被移除")

            # 保存原始数据
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"\n[完成] 保存原始数据: {output_file}")
            logger.info(f"       总记录数: {len(df):,}")

            # 清除断点文件
            clear_checkpoints(category, year)
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)

            return df
        else:
            logger.error(f"[错误] 未下载到任何数据: {category} - {year}")
            return None

    def download_all_categories(
        self,
        years: List[int] = None,
        categories: List[str] = None,
        force_redownload: bool = False
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        下载所有指定类别和年份的数据

        Args:
            years: 年份列表
            categories: 类别列表
            force_redownload: 是否强制重新下载

        Returns:
            嵌套字典: {category: {year: DataFrame}}
        """
        if years is None:
            years = list(range(START_YEAR, END_YEAR + 1))

        if categories is None:
            categories = list(CRIME_CATEGORIES.keys())

        self.download_stats["start_time"] = datetime.now()

        results = {}

        for category in categories:
            results[category] = {}

            for year in years:
                try:
                    df = self.download_year_category(category, year, force_redownload)
                    if df is not None:
                        results[category][year] = df
                        self.download_stats["total_records"] += len(df)
                except Exception as e:
                    logger.error(f"下载 {category} - {year} 时发生错误: {e}")

                # 年份之间等待
                if year < years[-1]:
                    logger.info(f"\n等待5秒后下载下一年份...")
                    time.sleep(5)

            # 类别之间等待
            if category != categories[-1]:
                logger.info(f"\n等待10秒后下载下一类别...")
                time.sleep(10)

        self.download_stats["end_time"] = datetime.now()

        return results

    def get_download_stats(self) -> Dict:
        """获取下载统计信息"""
        stats = self.download_stats.copy()

        if stats["start_time"] and stats["end_time"]:
            duration = (stats["end_time"] - stats["start_time"]).total_seconds()
            stats["duration_seconds"] = duration
            stats["duration_formatted"] = str(stats["end_time"] - stats["start_time"]).split('.')[0]
            stats["avg_records_per_second"] = stats["total_records"] / duration if duration > 0 else 0

        stats["success_rate"] = (
            stats["successful_requests"] / stats["total_requests"] * 100
            if stats["total_requests"] > 0 else 0
        )

        return stats


# ==================== 数据清洗类 ====================

class CrimeDataCleaner:
    """犯罪数据清洗器"""

    @staticmethod
    def clean_dataframe(df: pd.DataFrame, category: str, year: int) -> pd.DataFrame:
        """
        清洗犯罪数据DataFrame

        Args:
            df: 原始DataFrame
            category: 犯罪类别
            year: 年份

        Returns:
            清洗后的DataFrame
        """
        logger.info(f"\n清洗数据: {category} - {year}")
        original_count = len(df)
        logger.info(f"  原始记录数: {original_count:,}")

        # 1. 删除完全重复的行
        df = df.drop_duplicates()
        logger.info(f"  删除重复行后: {len(df):,} (-{original_count - len(df)})")

        # 2. 转换日期字段
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                before_drop = len(df)
                df = df.dropna(subset=['date'])
                logger.info(f"  转换日期后: {len(df):,} (-{before_drop - len(df)})")
            except Exception as e:
                logger.warning(f"  日期转换失败: {e}")

        # 3. 转换数值字段
        numeric_columns = [
            'beat', 'district', 'ward', 'community_area', 'year',
            'latitude', 'longitude', 'x_coordinate', 'y_coordinate'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. 删除经纬度缺失的记录（用于空间分析）
        before_geo = len(df)
        df_clean = df.dropna(subset=['latitude', 'longitude']).copy()
        logger.info(f"  删除缺失经纬度后: {len(df_clean):,} (-{before_geo - len(df_clean)})")

        # 5. 验证坐标范围（芝加哥大致范围）
        if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
            valid_lat = (df_clean['latitude'] >= 41.6) & (df_clean['latitude'] <= 42.1)
            valid_lon = (df_clean['longitude'] >= -88.0) & (df_clean['longitude'] <= -87.5)
            valid_coords = valid_lat & valid_lon

            if not valid_coords.all():
                invalid_count = (~valid_coords).sum()
                df_clean = df_clean[valid_coords].copy()
                logger.info(f"  删除无效坐标后: {len(df_clean):,} (-{invalid_count})")

        # 6. 添加标准化字段
        df_clean['category'] = category
        df_clean['category_name'] = CRIME_CATEGORIES[category]['name']

        # 7. 标准化primary_type（统一命名）
        type_mapping = {
            'CRIM SEXUAL ASSAULT': 'CRIMINAL SEXUAL ASSAULT',
            'OTHER NARCOTIC VIOLATION': 'NARCOTICS'
        }
        if 'primary_type' in df_clean.columns:
            # 确保 primary_type 是字符串类型
            try:
                df_clean['primary_type'] = df_clean['primary_type'].astype(str)
                df_clean['primary_type_std'] = df_clean['primary_type'].replace(type_mapping)
            except Exception as e:
                logger.warning(f"  primary_type 标准化失败: {e}")
                df_clean['primary_type_std'] = df_clean['primary_type']

        # 8. 添加时间特征
        if 'date' in df_clean.columns:
            df_clean['hour'] = df_clean['date'].dt.hour
            df_clean['day_of_week'] = df_clean['date'].dt.dayofweek  # 0=周一, 6=周日
            df_clean['month'] = df_clean['date'].dt.month
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6])

        # 9. 添加布尔字段转换
        for col in ['arrest', 'domestic']:
            if col in df_clean.columns:
                # 安全地转换为布尔值，处理可能的字典类型数据
                def safe_bool_convert(val):
                    if val is None:
                        return False
                    if isinstance(val, dict):
                        # 如果值是字典，尝试提取布尔值
                        return val.get('value', False) if isinstance(val.get('value'), bool) else False
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, str):
                        return val.lower() == 'true'
                    return bool(val)

                try:
                    df_clean[col] = df_clean[col].apply(safe_bool_convert)
                except Exception as e:
                    logger.warning(f"  布尔转换失败 {col}: {e}，使用默认值 False")
                    df_clean[col] = False

        # 10. 重置索引
        df_clean = df_clean.reset_index(drop=True)

        logger.info(f"[完成] 清洗完成: {len(df_clean):,} 条有效记录")

        return df_clean

    @staticmethod
    def save_cleaned_data(df: pd.DataFrame, category: str, year: int):
        """保存清洗后的数据"""
        output_file = os.path.join(CLEANED_DIR, f"{category}_{year}_cleaned.csv")
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"  保存清洗数据: {output_file}")


# ==================== 数据质量报告类 ====================

class DataQualityReporter:
    """数据质量报告生成器"""

    @staticmethod
    def generate_year_report(df: pd.DataFrame, category: str, year: int) -> Dict:
        """生成单年份数据质量报告"""
        report = {
            "category": category,
            "category_name": CRIME_CATEGORIES[category]['name'],
            "year": year,
            "total_records": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns and len(df) > 0 else None,
                "end": df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns and len(df) > 0 else None
            },
            "primary_type_distribution": df['primary_type'].value_counts().to_dict() if 'primary_type' in df.columns else {},
            "arrest_rate": float(df['arrest'].mean()) if 'arrest' in df.columns else None,
            "domestic_rate": float(df['domestic'].mean()) if 'domestic' in df.columns else None,
            "missing_data": {
                col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0
            },
            "spatial_coverage": {
                "latitude_range": [float(df['latitude'].min()), float(df['latitude'].max())] if 'latitude' in df.columns else None,
                "longitude_range": [float(df['longitude'].min()), float(df['longitude'].max())] if 'longitude' in df.columns else None
            }
        }

        return report

    @staticmethod
    def generate_summary_report(all_data: Dict[str, Dict[int, pd.DataFrame]]) -> Dict:
        """生成总体数据质量报告"""
        summary = {
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "time_period": f"{START_YEAR}-{END_YEAR}",
            "categories": {}
        }

        total_records = 0

        for category, year_data in all_data.items():
            if category not in CRIME_CATEGORIES:
                continue

            category_summary = {
                "name": CRIME_CATEGORIES[category]['name'],
                "description": CRIME_CATEGORIES[category]['description'],
                "total_records": 0,
                "years": {}
            }

            for year, df in year_data.items():
                if df is not None:
                    record_count = len(df)
                    category_summary['total_records'] += record_count
                    category_summary['years'][str(year)] = record_count
                    total_records += record_count

            summary['categories'][category] = category_summary

        # 计算总体统计
        summary['overall'] = {
            "total_records": total_records
        }

        for cat_key in CRIME_CATEGORIES.keys():
            if cat_key in summary['categories']:
                summary['overall'][f"{cat_key}_crimes"] = summary['categories'][cat_key]['total_records']

        return summary

    @staticmethod
    def save_report(report: Dict, filename: str):
        """保存报告到JSON文件"""
        filepath = os.path.join(REPORTS_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"  保存报告: {filepath}")


# ==================== 主函数 ====================

def main():
    """主函数"""

    # logger.info("="*70)
    # logger.info("           芝加哥犯罪数据下载系统 (2022-2023)")
    # logger.info("="*70)
    # logger.info("")
    # logger.info("犯罪类型分类:")
    # for cat_key, cat_info in CRIME_CATEGORIES.items():
    #     logger.info("")
    #     logger.info(f"  {cat_info['name']} ({cat_key}):")
    #     logger.info(f"    描述: {cat_info['description']}")
    #     logger.info(f"    包含 {len(cat_info['primary_types'])} 种:")
    #     for pt in cat_info['primary_types']:
    #         logger.info(f"      - {pt}")

    # logger.info("")
    # logger.info(f"时间范围: {START_YEAR} - {END_YEAR}")
    # logger.info(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    # logger.info("")
    # logger.info("="*70)

    # # 确认开始
    # try:
    #     response = input("\n是否开始下载？(y/n): ")
    #     if response.lower() != 'y':
    #         logger.info("取消下载")
    #         return
    # except (KeyboardInterrupt, EOFError):
    #     logger.info("\n取消下载")
    #     return

    # # 1. 下载数据
    # logger.info("\n[阶段 1/3] 下载原始数据...")
    # downloader = ChicagoCrimeDownloader()

    # try:
    #     all_data = downloader.download_all_categories(
    #         years=list(range(START_YEAR, END_YEAR + 1)),
    #         categories=list(CRIME_CATEGORIES.keys()),
    #         force_redownload=False
    #     )
    # except KeyboardInterrupt:
    #     logger.info("\n用户中断下载")
    #     return
    # except Exception as e:
    #     logger.error(f"下载过程中发生错误: {e}", exc_info=True)
    #     return

    # if not all_data or not any(all_data.values()):
    #     logger.error("没有成功下载任何数据，程序退出")
    #     return

    # 从已保存的CSV文件加载并清洗数据
    logger.info("\n[阶段 1/2] 从CSV文件加载并清洗数据...")
    cleaner = CrimeDataCleaner()

    all_cleaned_data = {}
    loaded_files = []

    for category in CRIME_CATEGORIES.keys():
        all_cleaned_data[category] = {}

        for year in range(START_YEAR, END_YEAR + 1):
            raw_file = os.path.join(RAW_DIR, f"{category}_{year}.csv")

            if not os.path.exists(raw_file):
                logger.warning(f"文件不存在，跳过: {raw_file}")
                continue

            try:
                logger.info(f"加载: {raw_file}")
                df = pd.read_csv(raw_file)
                loaded_files.append((category, year, len(df)))

                if len(df) > 0:
                    cleaned_df = cleaner.clean_dataframe(df, category, year)
                    cleaner.save_cleaned_data(cleaned_df, category, year)
                    all_cleaned_data[category][year] = cleaned_df
            except Exception as e:
                logger.error(f"清洗 {category} - {year} 时发生错误: {e}")

    if not loaded_files:
        logger.error("没有加载到任何数据文件，程序退出")
        return

    logger.info(f"\n成功加载 {len(loaded_files)} 个文件:")
    for cat, yr, cnt in loaded_files:
        logger.info(f"  {cat}_{yr}.csv: {cnt:,} 条")

    # 2. 生成报告
    logger.info("\n[阶段 2/2] 生成数据质量报告...")
    reporter = DataQualityReporter()

    # 生成详细报告
    for category, year_data in all_cleaned_data.items():
        for year, df in year_data.items():
            if df is not None and len(df) > 0:
                try:
                    report = reporter.generate_year_report(df, category, year)
                    reporter.save_report(report, f"{category}_{year}_report.json")
                except Exception as e:
                    logger.error(f"生成报告 {category} - {year} 时发生错误: {e}")

    # 生成总体报告
    try:
        summary_report = reporter.generate_summary_report(all_cleaned_data)
        reporter.save_report(summary_report, "summary_report.json")
    except Exception as e:
        logger.error(f"生成总体报告时发生错误: {e}")

    # 4. 显示总结
    logger.info("")
    logger.info("="*70)
    logger.info("                    处理完成！")
    logger.info("="*70)

    try:
        logger.info("")
        logger.info("总体统计:")
        total = summary_report['overall'].get('total_records', 0)
        logger.info(f"  总记录数: {total:,}")
        for cat_key in CRIME_CATEGORIES.keys():
            count = summary_report['overall'].get(f"{cat_key}_crimes", 0)
            logger.info(f"  {CRIME_CATEGORIES[cat_key]['name']}: {count:,}")
    except Exception as e:
        logger.warning(f"显示统计信息时出错: {e}")

    logger.info("")
    logger.info("输出文件:")
    logger.info(f"  原始数据: {os.path.abspath(RAW_DIR)}")
    logger.info(f"  清洗数据: {os.path.abspath(CLEANED_DIR)}")
    logger.info(f"  报告文件: {os.path.abspath(REPORTS_DIR)}")

    # 统计清洗后的数据
    total_cleaned = sum(
        len(df) for cat_data in all_cleaned_data.values()
        for df in cat_data.values() if df is not None
    )
    logger.info("")
    logger.info(f"清洗后总记录数: {total_cleaned:,}")

    logger.info("")
    logger.info("="*70)

    # 5. 保存合并文件（可选）
    try:
        save_merged = input("\n是否保存合并的CSV文件？(y/n): ")
        if save_merged.lower() == 'y':
            logger.info("\n正在合并文件...")

            all_dfs = []

            # 合并每个类别的数据
            for category in CRIME_CATEGORIES.keys():
                cat_dfs = []
                for year, df in all_cleaned_data.get(category, {}).items():
                    if df is not None:
                        cat_dfs.append(df)

                if cat_dfs:
                    cat_all = pd.concat(cat_dfs, ignore_index=True)
                    output_path = os.path.join(CLEANED_DIR, f"{category}_all_{START_YEAR}_{END_YEAR}.csv")
                    cat_all.to_csv(output_path, index=False)
                    logger.info(f"  保存 {CRIME_CATEGORIES[category]['name']} 合并文件: {len(cat_all):,} 条")
                    all_dfs.append(cat_all)

            # 合并所有犯罪
            if all_dfs:
                all_crimes = pd.concat(all_dfs, ignore_index=True)
                output_path = os.path.join(CLEANED_DIR, f"all_crimes_{START_YEAR}_{END_YEAR}.csv")
                all_crimes.to_csv(output_path, index=False)
                logger.info(f"  保存所有犯罪合并文件: {len(all_crimes):,} 条")
    except (KeyboardInterrupt, EOFError):
        logger.info("\n跳过合并文件")


# ==================== 命令行接口 ====================

def cli():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(
        description=f"芝加哥犯罪数据下载工具 ({START_YEAR}-{END_YEAR})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 下载所有数据
  python chicago_crime_downloader.py

  # 只下载暴力犯罪
  python chicago_crime_downloader.py --categories violent

  # 只下载2023年数据
  python chicago_crime_downloader.py --years 2023

  # 强制重新下载
  python chicago_crime_downloader.py --force
        """
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        choices=list(CRIME_CATEGORIES.keys()),
        help='指定要下载的犯罪类别（默认: 全部）'
    )

    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        help=f'指定要下载的年份（默认: {START_YEAR}-{END_YEAR}）'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新下载（覆盖已存在的文件）'
    )

    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='跳过数据清洗步骤'
    )

    args = parser.parse_args()

    # 验证年份
    if args.years:
        for year in args.years:
            if year < START_YEAR or year > END_YEAR:
                logger.error(f"错误: 年份 {year} 不在支持范围内 ({START_YEAR}-{END_YEAR})")
                sys.exit(1)

    # 1. 下载数据
    downloader = ChicagoCrimeDownloader()

    all_data = downloader.download_all_categories(
        years=args.years if args.years else list(range(START_YEAR, END_YEAR + 1)),
        categories=args.categories if args.categories else list(CRIME_CATEGORIES.keys()),
        force_redownload=args.force
    )

    if not all_data or not any(all_data.values()):
        logger.error("没有成功下载任何数据")
        sys.exit(1)

    # 2. 清洗数据
    if not args.no_clean:
        logger.info("\n清洗数据...")
        cleaner = CrimeDataCleaner()

        for category, year_data in all_data.items():
            for year, df in year_data.items():
                if df is not None and len(df) > 0:
                    try:
                        cleaned_df = cleaner.clean_dataframe(df, category, year)
                        cleaner.save_cleaned_data(cleaned_df, category, year)
                    except Exception as e:
                        logger.error(f"清洗 {category} - {year} 时发生错误: {e}")

    # 3. 生成报告
    logger.info("\n生成报告...")
    reporter = DataQualityReporter()

    try:
        summary_report = reporter.generate_summary_report(all_data)
        reporter.save_report(summary_report, "summary_report.json")
        logger.info(f"\n完成！总记录数: {summary_report['overall']['total_records']:,}")
    except Exception as e:
        logger.error(f"生成报告时发生错误: {e}")


# ==================== 执行 ====================

if __name__ == "__main__":
    # 检查依赖
    missing_deps = []

    try:
        import sodapy
    except ImportError:
        missing_deps.append("sodapy")

    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import requests
    except ImportError:
        missing_deps.append("requests")

    try:
        import urllib3
    except ImportError:
        missing_deps.append("urllib3")

    if missing_deps:
        logger.error(f"错误: 缺少以下依赖库: {', '.join(missing_deps)}")
        logger.error("请运行: pip install " + " ".join(missing_deps))
        sys.exit(1)

    # 运行主程序
    main()
