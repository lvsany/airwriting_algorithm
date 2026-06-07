"""
Pre-study 采集入口

用法:
  受控点触（贴纸标注）:
    python -m datasets.prestudy.runner --task tap --subject s01 --sticker-color black

  连续书写（空格键标注）:
    python -m datasets.prestudy.runner --task write --subject s01 --lighting normal --speed normal
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def main():
    parser = argparse.ArgumentParser(description='Pre-study 数据采集')
    parser.add_argument('--task',          required=True, choices=['tap', 'write'])
    parser.add_argument('--subject',       required=True, help='受试者编号，如 s01')
    parser.add_argument('--cam',           type=int, default=0)
    # tap 专用
    parser.add_argument('--sticker-color', default='green',
                        choices=['green', 'yellow', 'pink', 'blue', 'black'])
    # write 专用
    parser.add_argument('--lighting', default='normal',
                        choices=['normal', 'low', 'side'])
    parser.add_argument('--speed',    default='normal',
                        choices=['slow', 'normal', 'fast'])
    args = parser.parse_args()

    if args.task == 'tap':
        from datasets.prestudy.collect_tap import run_collect
        run_collect(subject=args.subject,
                    sticker_color=args.sticker_color,
                    cam_id=args.cam)
    else:
        from datasets.prestudy.collect_write import run_collect
        run_collect(subject=args.subject,
                    lighting=args.lighting,
                    speed=args.speed,
                    cam_id=args.cam)


if __name__ == '__main__':
    main()
