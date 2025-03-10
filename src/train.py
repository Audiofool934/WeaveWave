#!/usr/bin/env python3
# Copyright (c) 2023-2024 WeaveWave Team.
# All rights reserved.

"""
WeaveWave训练脚本，基于Facebook AudioCraft框架。
此脚本用于启动MusicGen-Style模型的训练。
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
import torch

# 确保audiocraft在路径中
sys.path.append('audiocraft')

# 导入必要的工具
from audiocraft.environment import AudioCraftEnvironment
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ['AUDIOCRAFT_TEAM'] = 'default'
os.environ['DORA_DIR_ROOT'] = './outputs'  # 保存模型和日志的目录


def setup_environment():
    """设置训练环境"""
    # 初始化AudioCraft环境
    logger.info("初始化AudioCraft环境...")
    # 创建临时目录作为参考目录
    ref_dir = Path(tempfile.mkdtemp())
    os.environ['AUDIOCRAFT_REFERENCE_DIR'] = str(ref_dir)
    
    # 确保输出目录存在
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，训练可能会很慢!")
    else:
        logger.info(f"检测到 {torch.cuda.device_count()} 个GPU")


def train_musicgen_style():
    """使用dora启动MusicGen-Style训练任务"""
    logger.info("准备启动MusicGen-Style训练...")
    
    # 创建数据目录
    data_dir = Path('./data/multimodal_music_dataset')
    for split in ['train', 'valid', 'test']:
        (data_dir / split).mkdir(parents=True, exist_ok=True)
    
    # 构建dora命令
    config_path = "config/musicgen_style_32khz.yaml"
    if not Path(config_path).exists():
        logger.error(f"配置文件不存在: {config_path}")
        return
    
    # 使用hydra加载配置
    logger.info(f"加载配置文件: {config_path}")
    cfg = OmegaConf.load(config_path)
    logger.info("配置加载成功")
    
    # 导入训练模块
    logger.info("导入训练模块...")
    from audiocraft.train import main as train_main
    
    # 使用AudioCraft的训练入口点运行训练
    logger.info("启动训练...")
    try:
        # 将配置传递给训练函数
        train_main(cfg)
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """主函数"""
    logger.info("=== WeaveWave MusicGen-Style 训练启动 ===")
    setup_environment()
    train_musicgen_style()
    logger.info("=== 训练脚本执行完毕 ===")


if __name__ == "__main__":
    main() 