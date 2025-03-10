#!/usr/bin/env python3
# Copyright (c) 2023-2024 WeaveWave Team.
# All rights reserved.

"""
WeaveWave训练启动脚本
提供简单的命令行接口启动训练流程
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """设置基本环境"""
    # 确保config目录存在
    config_dir = Path('./config')
    if not config_dir.exists():
        logger.error("配置目录不存在，请确保您在正确的工作目录中")
        sys.exit(1)
    
    # 确保输出目录存在
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)
    
    # 确保数据目录存在
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    return True


def prepare_data(args):
    """准备数据集"""
    logger.info("准备数据集...")
    
    # 检查数据集是否已存在
    dataset_dir = Path('./data/multimodal_music_dataset')
    if dataset_dir.exists() and not args.force_rebuild_dataset:
        sample_count = sum(1 for _ in dataset_dir.glob('*/*.json'))
        logger.info(f"数据集已存在，包含 {sample_count} 个样本")
        if sample_count > 0 and input("是否继续使用现有数据集? [y/N]: ").lower() == 'y':
            return True
    
    # 运行数据准备脚本
    import subprocess
    
    cmd = [
        sys.executable, 'prepare_dataset.py',
        '--output_dir', './data/multimodal_music_dataset',
    ]
    
    if args.dummy_data:
        cmd.extend(['--create_dummy', '--dummy_samples', str(args.dummy_samples)])
    elif args.source_data:
        cmd.extend(['--source_dir', args.source_data])
    else:
        logger.warning("未指定数据源，将创建虚拟数据集用于测试")
        cmd.extend(['--create_dummy', '--dummy_samples', str(args.dummy_samples)])
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("数据准备完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"数据准备失败: {e}")
        return False


def run_training(args):
    """运行训练流程"""
    logger.info("开始训练流程...")
    
    # 运行训练脚本
    import subprocess
    
    # 基本训练命令
    cmd = [sys.executable, 'train.py']
    
    # 添加GPU设备参数
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
        logger.info(f"使用GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("训练完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"训练失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WeaveWave 训练启动脚本")
    
    # 数据相关参数
    data_group = parser.add_argument_group("数据参数")
    data_group.add_argument("--dummy_data", action="store_true",
                         help="使用虚拟数据集进行训练")
    data_group.add_argument("--dummy_samples", type=int, default=100,
                         help="虚拟数据集中的样本数量")
    data_group.add_argument("--source_data", type=str, default=None,
                         help="真实数据集的源目录")
    data_group.add_argument("--force_rebuild_dataset", action="store_true",
                         help="强制重建数据集，即使它已存在")
    
    # 训练相关参数
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument("--gpu", type=int, nargs='+', default=None,
                          help="要使用的GPU设备ID，例如 0 1 2 3")
    
    args = parser.parse_args()
    
    logger.info("=== WeaveWave 训练启动脚本 ===")
    
    # 设置环境
    if not setup_environment():
        return
    
    # 准备数据
    if not prepare_data(args):
        return
    
    # 运行训练
    run_training(args)
    
    logger.info("=== 训练流程结束 ===")


if __name__ == "__main__":
    main() 