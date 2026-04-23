"""
Exp-A 完整流程入口
包含三个子实验：
  Exp-A1: 纯 tap 数据采集与标注（荧光贴纸 HSV 自动标注 + 键盘备选）
  Exp-A2: 运动学 + 外观特征统计分析（时序曲线）
  Exp-A3: 各特征组合 ROC 曲线对比

用法:
  采集数据:   python exp_a_runner.py --mode collect --subject s01
  分析特征:   python exp_a_runner.py --mode analyze --subject s01
  ROC 对比:   python exp_a_runner.py --mode roc --subject s01
  全流程:     python exp_a_runner.py --mode all --subject s01
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from experiments.exp_a.a1_collect import run_collect
from experiments.exp_a.a2_analyze import run_analyze
from experiments.exp_a.a3_roc import run_roc


def parse_args():
    parser = argparse.ArgumentParser(description='Exp-A: 纯接触检测实验')
    parser.add_argument('--mode', choices=['collect', 'analyze', 'roc', 'all'],
                        default='all', help='运行模式')
    parser.add_argument('--subject', type=str, default='s01',
                        help='受试者 ID，用于文件命名（如 s01, s02）')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data'),
                        help='数据存储目录')
    parser.add_argument('--sticker-color', type=str, default='green',
                        choices=['green', 'yellow', 'pink', 'blue'],
                        help='荧光贴纸颜色（用于 HSV 自动标注）')
    parser.add_argument('--label-mode', type=str, default='sticker',
                        choices=['sticker', 'keyboard'],
                        help='标注方式：sticker=贴纸自动检测，keyboard=空格键手动')
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Exp-A  受试者: {args.subject}  标注: {args.label_mode}")
    print(f"  数据目录: {data_dir}")
    print(f"{'='*50}\n")

    if args.mode in ('collect', 'all'):
        print("[Exp-A1] 开始数据采集...")
        run_collect(subject=args.subject, data_dir=data_dir,
                    label_mode=args.label_mode,
                    sticker_color=args.sticker_color)
        print("[Exp-A1] 完成\n")

    if args.mode in ('analyze', 'all'):
        print("[Exp-A2] 开始特征统计分析...")
        run_analyze(subject=args.subject, data_dir=data_dir)
        print("[Exp-A2] 完成\n")

    if args.mode in ('roc', 'all'):
        print("[Exp-A3] 开始 ROC 曲线对比...")
        run_roc(subject=args.subject, data_dir=data_dir)
        print("[Exp-A3] 完成\n")


if __name__ == '__main__':
    main()
