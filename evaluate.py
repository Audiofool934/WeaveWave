#!/usr/bin/env python3
# Copyright (c) 2023-2024 WeaveWave Team.
# All rights reserved.

"""
WeaveWave评估脚本
用于评估训练好的模型性能
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class MusicGenStyleEvaluator:
    """MusicGen-Style模型评估器"""
    
    def __init__(self, model_path, device=None):
        """
        初始化评估器
        
        参数:
            model_path: 模型路径
            device: 运行设备，如果为None则自动选择
        """
        self.model_path = Path(model_path)
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"评估器初始化，模型路径: {self.model_path}")
        logger.info(f"使用设备: {self.device}")
        
        # 加载AudioCraft模型
        try:
            sys.path.append('audiocraft')
            from audiocraft.models import MusicGen
            self.MusicGen = MusicGen
            logger.info("成功导入MusicGen模型")
        except ImportError:
            logger.error("无法导入MusicGen，请确保audiocraft已正确安装")
            sys.exit(1)
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        logger.info("加载模型...")
        try:
            if self.model_path.exists():
                # 使用本地模型
                self.model = self.MusicGen.get_pretrained(str(self.model_path))
                logger.info("成功加载本地模型")
            else:
                # 使用预训练模型
                logger.warning(f"本地模型路径 {self.model_path} 不存在，使用预训练模型 'facebook/musicgen-style' 作为备用")
                self.model = self.MusicGen.get_pretrained('facebook/musicgen-style')
            
            # 设置模型参数
            self.model.set_generation_params(
                duration=10,  # 生成10秒的音频
                use_sampling=True,
                top_k=250,
                top_p=0.0,
                cfg_coef=3.0,
                cfg_coef_beta=5.0
            )
            
            # 设置风格条件器参数
            self.model.set_style_conditioner_params(
                eval_q=1,  # 量化级别
                excerpt_length=3.0,  # 摘录长度（秒）
            )
            
            # 移动模型到指定设备
            self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            sys.exit(1)
    
    def evaluate_text_to_music(self, texts, output_dir):
        """
        评估文本到音乐生成
        
        参数:
            texts: 文本描述列表
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"评估文本到音乐生成，样本数: {len(texts)}")
        
        # 生成音乐
        with torch.no_grad():
            wav = self.model.generate(texts)
        
        # 保存生成的音频
        for i, one_wav in enumerate(wav):
            output_path = output_dir / f"text2music_{i+1}.wav"
            torchaudio.save(
                str(output_path),
                one_wav.cpu(),
                self.model.sample_rate
            )
            logger.info(f"保存音频到: {output_path}")
        
        logger.info("文本到音乐生成评估完成")
    
    def evaluate_style_to_music(self, audio_paths, output_dir):
        """
        评估风格到音乐生成
        
        参数:
            audio_paths: 音频文件路径列表
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"评估风格到音乐生成，样本数: {len(audio_paths)}")
        
        for i, audio_path in enumerate(audio_paths):
            try:
                # 加载音频
                melody, sr = torchaudio.load(audio_path)
                
                # 确保格式正确
                if melody.dim() == 1:
                    melody = melody.unsqueeze(0)
                
                # 生成音乐
                with torch.no_grad():
                    wav = self.model.generate_with_chroma(
                        descriptions=[None],  # 无文本描述
                        melody=melody.to(self.device),
                        melody_sample_rate=sr,
                    )
                
                # 保存生成的音频
                output_path = output_dir / f"style2music_{i+1}.wav"
                torchaudio.save(
                    str(output_path),
                    wav[0].cpu(),
                    self.model.sample_rate
                )
                logger.info(f"保存音频到: {output_path}")
                
            except Exception as e:
                logger.error(f"处理音频 {audio_path} 时出错: {e}")
        
        logger.info("风格到音乐生成评估完成")
    
    def evaluate_style_and_text_to_music(self, texts, audio_paths, output_dir):
        """
        评估风格和文本到音乐生成
        
        参数:
            texts: 文本描述列表
            audio_paths: 音频文件路径列表
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 确保文本和音频数量相同
        assert len(texts) == len(audio_paths), "文本和音频数量必须相同"
        
        logger.info(f"评估风格和文本到音乐生成，样本数: {len(texts)}")
        
        for i, (text, audio_path) in enumerate(zip(texts, audio_paths)):
            try:
                # 加载音频
                melody, sr = torchaudio.load(audio_path)
                
                # 确保格式正确
                if melody.dim() == 1:
                    melody = melody.unsqueeze(0)
                
                # 生成音乐
                with torch.no_grad():
                    wav = self.model.generate_with_chroma(
                        descriptions=[text],
                        melody=melody.to(self.device),
                        melody_sample_rate=sr,
                    )
                
                # 保存生成的音频
                output_path = output_dir / f"style_and_text2music_{i+1}.wav"
                torchaudio.save(
                    str(output_path),
                    wav[0].cpu(),
                    self.model.sample_rate
                )
                logger.info(f"保存音频到: {output_path}")
                
            except Exception as e:
                logger.error(f"处理样本 {i+1} 时出错: {e}")
        
        logger.info("风格和文本到音乐生成评估完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="WeaveWave 评估脚本")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="./outputs/latest_model",
                      help="模型路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation",
                      help="评估结果输出目录")
    parser.add_argument("--gpu", type=int, default=None,
                      help="GPU设备ID")
    
    # 评估模式
    eval_group = parser.add_argument_group("评估模式")
    eval_group.add_argument("--eval_text2music", action="store_true",
                         help="评估文本到音乐生成")
    eval_group.add_argument("--eval_style2music", action="store_true",
                         help="评估风格到音乐生成")
    eval_group.add_argument("--eval_style_and_text2music", action="store_true",
                         help="评估风格和文本到音乐生成")
    
    # 评估数据
    data_group = parser.add_argument_group("评估数据")
    data_group.add_argument("--text_file", type=str, default=None,
                         help="包含文本描述的文件，每行一个描述")
    data_group.add_argument("--audio_dir", type=str, default=None,
                         help="包含音频文件的目录")
    
    args = parser.parse_args()
    
    logger.info("=== WeaveWave 评估脚本 ===")
    
    # 设置设备
    device = None
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = 'cuda'
        logger.info(f"使用GPU: {args.gpu}")
    
    # 创建评估器
    evaluator = MusicGenStyleEvaluator(args.model_path, device)
    
    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载文本和音频数据
    texts = []
    if args.text_file:
        with open(args.text_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"从 {args.text_file} 加载了 {len(texts)} 个文本描述")
    else:
        # 使用默认文本
        texts = [
            "Upbeat electronic dance music with energetic synthesizers",
            "Calm piano melody with gentle strings in the background",
            "Heavy rock with distorted guitars and powerful drums",
            "Jazz fusion with smooth saxophone solo and walking bass line",
            "Orchestral cinematic music with epic brass section"
        ]
        logger.info(f"使用 {len(texts)} 个默认文本描述")
    
    audio_paths = []
    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        if audio_dir.exists():
            audio_paths = list(audio_dir.glob('*.wav')) + list(audio_dir.glob('*.mp3'))
            logger.info(f"从 {args.audio_dir} 加载了 {len(audio_paths)} 个音频文件")
    
    if not audio_paths and (args.eval_style2music or args.eval_style_and_text2music):
        logger.error("需要音频文件进行风格评估，请使用 --audio_dir 指定音频目录")
        sys.exit(1)
    
    # 执行评估
    if args.eval_text2music:
        evaluator.evaluate_text_to_music(texts, output_dir / 'text2music')
    
    if args.eval_style2music and audio_paths:
        evaluator.evaluate_style_to_music(audio_paths, output_dir / 'style2music')
    
    if args.eval_style_and_text2music and audio_paths:
        # 确保文本和音频数量相同
        if len(texts) >= len(audio_paths):
            texts = texts[:len(audio_paths)]
        else:
            texts = texts * (len(audio_paths) // len(texts) + 1)
            texts = texts[:len(audio_paths)]
        
        evaluator.evaluate_style_and_text_to_music(texts, audio_paths, output_dir / 'style_and_text2music')
    
    logger.info("=== 评估完成 ===")


if __name__ == "__main__":
    main() 