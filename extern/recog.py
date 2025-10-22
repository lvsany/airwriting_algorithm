"""
手写文字识别模块

使用QwenVL API进行手写数字和英文单词的识别，支持批量处理和准确率统计。
"""

import os
import base64
import json
import re
import csv
from typing import Dict, List, Tuple
from pathlib import Path

import yaml
import requests


# =============================================================================
# 配置加载
# =============================================================================

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as file:
    CONFIG = yaml.safe_load(file)


# =============================================================================
# 工具函数
# =============================================================================

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    计算字符错误率 (Character Error Rate)
    
    使用编辑距离（Levenshtein距离）算法计算识别结果与真实标签之间的差异。
    
    Args:
        reference: 真实标签
        hypothesis: 识别结果
        
    Returns:
        CER值 (0.0 - 1.0)，0表示完全正确，1表示完全错误
    """
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    
    len_ref = len(reference)
    len_hyp = len(hypothesis)
    
    # 创建距离矩阵
    d = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]
    
    # 初始化边界
    for i in range(len_ref + 1):
        d[i][0] = i
    for j in range(len_hyp + 1):
        d[0][j] = j
    
    # 动态规划计算编辑距离
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # 删除
                d[i][j-1] + 1,      # 插入
                d[i-1][j-1] + cost  # 替换
            )
    
    edit_distance = d[len_ref][len_hyp]
    cer = edit_distance / len_ref
    
    return cer


def extract_ground_truth_from_filename(filename: str) -> Tuple[str, str]:
    """
    从文件名中提取真实标签和内容类型
    
    Args:
        filename: 文件名，格式如 "cly_1234.mp4" 或 "cly_apple.mp4"
        
    Returns:
        (label, content_type): 提取出的标签和内容类型("digit" 或 "english")
    """
    # 移除扩展名
    name = os.path.splitext(filename)[0]
    
    # 按下划线分割，取最后一个部分作为标签
    parts = name.split('_')
    if len(parts) < 2:
        return "", "digit"
    
    label = parts[-1]
    
    # 判断是数字还是英文
    if label.isdigit():
        return label, "digit"
    elif label.isalpha():
        return label.lower(), "english"
    else:
        # 包含数字和字母的情况，优先查找纯数字
        numbers = re.findall(r'\d+', label)
        if numbers:
            return max(numbers, key=len), "digit"
        # 查找纯字母
        words = re.findall(r'[a-zA-Z]+', label)
        if words:
            return max(words, key=len).lower(), "english"
    
    return "", "digit"


def get_prompt_for_image_type(image_type: str, content_type: str = "digit") -> str:
    """
    根据图片类型和内容类型返回相应的提示词
    
    Args:
        image_type: 图片类型 ("no_smoothing", "plain", "segmented", "projection", "myself")
        content_type: 内容类型 ("digit" 表示数字, "english" 表示英文单词)
        
    Returns:
        对应的提示词
    """
    if content_type == "english":
        prompts = {
            "no_smoothing": "这是一张未经平滑处理的手写英文单词轨迹图。图片中包含原始的、可能有抖动和噪声的手写英文单词轨迹。请仔细识别图中的连续手写英文单词，只返回英文单词本身，不要其他文字。例如：apple",
            
            "plain": "这是一张经过平滑处理的手写英文单词轨迹图。图片中的轨迹相对平滑，噪声较少。请识别图中的连续手写英文单词，只返回英文单词本身，不要其他文字。例如：apple",
            
            "segmented": "这是一张经过轨迹分割处理的手写英文单词图。图片中可能包含不同颜色的线段和虚线，虚线通常表示字母之间的连接部分或笔画间的过渡。请重点关注实线部分识别手写英文单词，虚线部分可能是字母间的连接，请慎重考虑后识别。只返回英文单词本身，不要其他文字。例如：apple",
            
            "projection": "这是一张纯净版投影分割图，包含手写英文单词轨迹和红色分界线。红色虚线标记了字母之间的边界。请根据轨迹识别手写英文单词，只返回英文单词本身，不要其他文字。例如：apple",

            "myself": "这是一张用户自定义的手写英文单词轨迹图。请根据图中的轨迹识别手写英文单词，只返回英文单词本身，不要其他文字。例如：apple",
        }
    else:  # digit
        prompts = {
            "no_smoothing": "这是一张未经平滑处理的手写数字轨迹图。图片中包含原始的、可能有抖动和噪声的手写数字轨迹。请仔细识别图中的连续四个手写数字，必须且只能返回4个数字，不要其他文字。输出格式：XXXX（4个数字），例如：1234",
            
            "plain": "这是一张经过平滑处理的手写数字轨迹图。图片中的轨迹相对平滑，噪声较少。请识别图中的连续四个手写数字，必须且只能返回4个数字，不要其他文字。输出格式：XXXX（4个数字），例如：1234",

            "segmented": "这是一张经过轨迹分割处理的连续四个数字的手写数字图。图片中可能包含不同颜色的线段和虚线，虚线通常表示数字间的连接（但也不排除他是字符的比划的一部分）。不能保证把四个数字完全分割开，在识别的过程中要考虑人类书写习惯。请识别图中的四个手写数字，必须且只能返回4个数字，不多不少。输出格式：XXXX（4个数字），例如：1234",
            
            "projection": "这是一张纯净版投影分割图，包含手写数字轨迹和红色分界线。红色虚线标记了数字之间的边界，将连笔轨迹分成不同的数字。请按照轨迹顺序，根据红色分界线独立识别每个区域的数字，必须且只能返回4个数字。输出格式：XXXX（4个数字），例如：1234",

            "myself": "这是一张用户自定义的手写数字轨迹图。请根据图中的轨迹识别手写数字，必须且只能返回4个数字，不要其他文字。输出格式：XXXX（4个数字），例如：1234",
        }
    
    return prompts.get(image_type, prompts["plain"])


# =============================================================================
# QwenVL识别器类
# =============================================================================


class QwenVLRecognizer:
    """
    使用QwenVL API进行手写文字识别的类
    
    支持手写数字和英文单词的识别。
    """
    
    def __init__(self, api_key: str, api_url: str = None):
        """
        初始化QwenVL识别器
        
        Args:
            api_key: QwenVL API密钥
            api_url: API端点URL (如果不提供将使用默认)
        """
        self.api_key = api_key
        self.api_url = api_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        将图片编码为base64字符串
        
        Args:
            image_path: 图片路径
            
        Returns:
            base64编码的图片字符串
        """
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def recognize_text(self, image_path: str, prompt: str = None, content_type: str = "digit") -> str:
        """
        识别单张图片中的手写文字（数字或英文单词）
        
        Args:
            image_path: 图片路径
            prompt: 提示词，如果不提供将使用默认
            content_type: 内容类型，"digit" 表示数字，"english" 表示英文单词
            
        Returns:
            识别出的文字字符串
        """
        result = self.recognize_text_with_confidence(image_path, prompt, content_type)
        return result['text'] if result else ""
    
    def recognize_text_with_confidence(self, image_path: str, prompt: str = None, content_type: str = "digit") -> Dict:
        """
        识别单张图片中的手写文字，并返回置信度信息
        
        Args:
            image_path: 图片路径
            prompt: 提示词，如果不提供将使用默认
            content_type: 内容类型，"digit" 表示数字，"english" 表示英文单词
            
        Returns:
            包含识别结果和置信度的字典:
            {
                'text': '识别的文字',
                'raw_response': '原始响应',
                'digit_confidences': [0.9, 0.8, 0.95, 0.7],  # 仅数字识别
                'confidence_based_result': '1234'  # 基于置信度的结果
            }
        """
        if prompt is None:
            if content_type == "english":
                prompt = "图片中是手写的英文单词轨迹，请识别图片中的手写英文单词，只返回英文单词本身，不要其他文字。"
            else:
                # 修改提示词，要求输出置信度
                prompt = """图片中是四个数字连笔手写的轨迹，请识别图片中的手写数字。

要求：
1. 必须识别出4个数字
2. 为每个数字提供置信度评分（0-1之间的小数）
3. 输出格式必须严格按照JSON格式：
{
  "digits": [
    {"digit": "1", "confidence": 0.95},
    {"digit": "2", "confidence": 0.88},
    {"digit": "3", "confidence": 0.92},
    {"digit": "4", "confidence": 0.76}
  ]
}

注意事项：
- 如果图片中的轨迹存在虚线，虚线可能是两个数字之间的间隔，请慎重考虑
- confidence越高表示越确定该数字的识别结果
- 必须返回有效的JSON格式"""
        
        try:
            # 编码图片
            image_base64 = self.encode_image_to_base64(image_path)
            
            # 构造请求数据
            data = {
                "model": "qwen-vl-max",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": f"data:image/png;base64,{image_base64}"
                                },
                                {
                                    "type": "text", 
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                },
                "parameters": {
                    "temperature": 0.1,
                    "max_tokens": 300  # 增加token限制以支持JSON输出
                }
            }
            
            # 发送请求
            timeout = CONFIG['text_recognition']['api_timeout']
            response = requests.post(self.api_url, headers=self.headers, json=data, timeout=timeout)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            if 'output' not in result or 'choices' not in result['output']:
                print(f"API响应格式异常: {result}")
                return {'text': '', 'raw_response': str(result)}
            
            content = result['output']['choices'][0]['message']['content']
            
            # 处理不同的content格式
            recognition_text = ""
            if isinstance(content, str):
                recognition_text = content
            elif isinstance(content, list):
                # 从列表中提取所有text字段
                text_parts = [item['text'] for item in content if isinstance(item, dict) and 'text' in item]
                recognition_text = ''.join(text_parts)
            else:
                print(f"API返回内容类型异常: {type(content)}, 内容: {content}")
                return {'text': '', 'raw_response': str(content)}
            
            # 根据内容类型提取结果
            if content_type == "english":
                # 提取英文单词（只保留字母）
                words = re.findall(r'[a-zA-Z]+', recognition_text)
                result_text = words[0].lower() if words else ""
                return {
                    'text': result_text,
                    'raw_response': recognition_text
                }
            else:
                # 数字识别：尝试解析JSON格式的置信度信息
                return self._parse_digit_response_with_confidence(recognition_text)
                
        except requests.exceptions.RequestException as e:
            print(f"API请求失败 {image_path}: {e}")
            return {'text': '', 'error': str(e)}
        except json.JSONDecodeError as e:
            print(f"JSON解析失败 {image_path}: {e}")
            return {'text': '', 'error': str(e)}
        except KeyError as e:
            print(f"响应格式错误 {image_path}: 缺少键 {e}")
            return {'text': '', 'error': str(e)}
        except Exception as e:
            print(f"识别图片 {image_path} 时出错: {e}")
            return {'text': '', 'error': str(e)}
    
    def _parse_digit_response_with_confidence(self, response_text: str) -> Dict:
        """
        解析包含置信度的数字识别响应
        
        Args:
            response_text: API返回的文本
            
        Returns:
            包含识别结果和置信度的字典
        """
        # 尝试提取JSON部分
        json_match = re.search(r'\{[^}]*"digits"[^}]*\}', response_text, re.DOTALL)
        
        if json_match:
            try:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                if 'digits' in parsed and isinstance(parsed['digits'], list):
                    digits = []
                    confidences = []
                    
                    for item in parsed['digits']:
                        if isinstance(item, dict) and 'digit' in item:
                            digit = str(item['digit'])
                            confidence = float(item.get('confidence', 0.5))
                            digits.append(digit)
                            confidences.append(confidence)
                    
                    result_text = ''.join(digits)
                    
                    # 基于置信度的结果：只保留置信度>=0.5的数字
                    confidence_based = ''.join([d for d, c in zip(digits, confidences) if c >= 0.5])
                    
                    return {
                        'text': result_text,
                        'raw_response': response_text,
                        'digit_confidences': confidences,
                        'confidence_based_result': confidence_based,
                        'parsed_json': parsed
                    }
            except (json.JSONDecodeError, ValueError) as e:
                print(f"JSON解析失败: {e}, 内容: {json_str}")
        
        # 如果无法解析JSON，回退到简单提取数字
        digits = re.findall(r'\d+', response_text)
        result_text = ''.join(digits) if digits else ""
        
        return {
            'text': result_text,
            'raw_response': response_text,
            'digit_confidences': [],
            'confidence_based_result': result_text
        }
    
    def recognize_digit(self, image_path: str, prompt: str = None) -> str:
        """
        识别单张图片中的手写数字（向后兼容方法）
        
        Args:
            image_path: 图片路径
            prompt: 提示词，如果不提供将使用默认
            
        Returns:
            识别出的数字字符串
        """
        return self.recognize_text(image_path, prompt, "digit")


# =============================================================================
# 批量识别函数
# =============================================================================


def batch_recognize_text(results_dir: str, api_key: str, 
                        output_csv: str = "data/results/recognition_results.csv") -> Dict[str, Dict]:
    """
    批量识别results目录下的手写文字图片（数字或英文单词）
    
    Args:
        results_dir: results目录路径，包含no_smoothing/, plain/, segmented/, projection/等子目录
        api_key: QwenVL API密钥
        output_csv: 输出CSV文件路径
        
    Returns:
        包含识别结果和准确率的字典
    """
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return {}
    
    recognizer = QwenVLRecognizer(api_key)
    
    # 初始化结果字典
    results = {
        "no_smoothing": [],
        "plain": [],
        "segmented": [],
        "projection": [],
        "clean_projection": [],
        "myself": []
    }
    
    # 处理各个子目录
    subdirs = list(results.keys())
    
    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.exists(subdir_path):
            continue
            
        # 获取该子目录下的所有PNG文件
        png_files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.png')])
        if not png_files:
            continue
            
        print(f"\n【{subdir}】({len(png_files)} 个图片)")
        
        for png_file in png_files:
            # 从文件名提取真实标签和内容类型
            ground_truth, content_type = extract_ground_truth_from_filename(png_file)
            if not ground_truth:
                continue
                
            img_path = os.path.join(subdir_path, png_file)
            
            # 根据图片类型和内容类型获取相应的提示词
            prompt = get_prompt_for_image_type(subdir, content_type)
            
            # 使用带置信度的识别方法
            recog_result = recognizer.recognize_text_with_confidence(img_path, prompt, content_type)
            recognized = recog_result.get('text', '')
            
            # 计算CER
            cer = calculate_cer(ground_truth, recognized)
            
            # 如果有置信度信息，也计算基于置信度的结果的CER
            confidence_based = recog_result.get('confidence_based_result', '')
            cer_confidence = calculate_cer(ground_truth, confidence_based) if confidence_based else cer
            
            # 保存结果
            result_entry = {
                'filename': png_file,
                'image_path': img_path,
                'ground_truth': ground_truth,
                'recognized': recognized,
                'content_type': content_type,
                'correct': ground_truth == recognized,
                'cer': cer
            }
            
            # 添加置信度相关信息（仅数字识别）
            if content_type == 'digit' and recog_result.get('digit_confidences'):
                result_entry['digit_confidences'] = recog_result['digit_confidences']
                result_entry['confidence_based_result'] = confidence_based
                result_entry['confidence_based_correct'] = ground_truth == confidence_based
                result_entry['cer_confidence'] = cer_confidence
            
            results[subdir].append(result_entry)
            
            # 输出识别结果
            status = "✓" if ground_truth == recognized else "✗"
            confidence_info = ""
            if content_type == 'digit' and recog_result.get('digit_confidences'):
                confidences = recog_result['digit_confidences']
                conf_str = ', '.join([f"{c:.2f}" for c in confidences])
                confidence_info = f" | 置信度=[{conf_str}]"
                if confidence_based and confidence_based != recognized:
                    conf_status = "✓" if ground_truth == confidence_based else "✗"
                    confidence_info += f" | 置信度结果={confidence_based} {conf_status}"
            
            print(f"  {png_file}: GT={ground_truth} | 识别={recognized} | {status}{confidence_info}")
        # —— 在统计前，构建并追加 mixed 结果 ——
    grouped_by_name = _group_results_by_basename(results)
    mixed_entries = _build_mixed_results(grouped_by_name)
    if mixed_entries:
        results['mixed'] = mixed_entries
        print(f"\n【mixed】({len(mixed_entries)} 个图片)")
        for r in mixed_entries:
            gt = r['ground_truth']
            rec = r['recognized']
            status = "✓" if r['correct'] else "✗"
            if r.get('content_type') == 'digit' and r.get('digit_confidences'):
                conf_str = ', '.join([f"{c:.2f}" for c in r['digit_confidences']])
                print(f"  {r['filename']}: GT={gt} | 融合识别={rec} | {status} | 逐位置信度=[{conf_str}]")
            else:
                print(f"  {r['filename']}: GT={gt} | 融合识别={rec} | {status}")

    # 计算并打印统计结果
    accuracy_results = _calculate_accuracy_statistics(results)
    
    # 保存详细结果到CSV
    save_results_to_csv(results, output_csv)
    
    return {
        'detailed_results': results,
        'accuracy_summary': accuracy_results
    }


def batch_recognize_digits(results_dir: str, api_key: str, 
                          output_csv: str = "recognition_results.csv") -> Dict[str, Dict]:
    """
    批量识别results目录下的手写数字图片（向后兼容函数）
    
    Args:
        results_dir: results目录路径
        api_key: QwenVL API密钥
        output_csv: 输出CSV文件路径
        
    Returns:
        包含识别结果和准确率的字典
    """
    return batch_recognize_text(results_dir, api_key, output_csv)


def _group_results_by_basename(results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, Dict]]:
    """
    将各个子目录识别结果按 '同名文件'(basename) 分组，便于做 mixed 融合。
    返回结构：{ basename: { img_type: result_entry, ... }, ... }
    """
    grouped = {}
    for img_type, lst in results.items():
        if not lst or img_type == 'mixed':
            continue
        for r in lst:
            basename = os.path.basename(r['filename'])
            grouped.setdefault(basename, {})[img_type] = r
    return grouped


def _build_mixed_results(grouped: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    """
    针对同名图片在 (no_smoothing, plain, projection, segmented) 四个版本里
    的识别结果进行融合：
      - 若为数字：逐字符选择“置信度最高”的候选；若缺置信度则用 0.5 兜底
      - 若为英文：用“多数投票”（出现频次最高），若并列则取最短的
    返回：mixed 的 result_entry 列表，结构与其他类型一致
    """
    preferred_order = ['projection', 'segmented', 'plain', 'no_smoothing']  # 置信度并列时的优先顺序
    mixed_entries = []

    for basename, per_type in grouped.items():
        # 推断 GT / content_type / image_path（任选一个存在的类型）
        any_entry = next(iter(per_type.values()))
        ground_truth = any_entry.get('ground_truth', '')
        content_type = any_entry.get('content_type', 'digit')
        img_path = any_entry.get('image_path', '')
        fname = any_entry.get('filename', basename)

        # 仅融合我们关心的几类
        candidates = {k: v for k, v in per_type.items()
                      if k in ('no_smoothing', 'plain', 'projection', 'segmented')}

        if not candidates:
            # 没有可融合的来源，跳过
            continue

        if content_type == 'digit':
            # 目标长度（默认 4），从 GT 推断
            target_len = len(ground_truth) if ground_truth else 4

            # 收集各来源的识别串与逐位置信度
            per_source_digits = {}
            for t, r in candidates.items():
                text = str(r.get('recognized', '') or '')
                confs = r.get('digit_confidences') or []
                # 若长度不匹配，用简单修复：不够则右侧补 0 置信度=0.5；过长则截断
                digits = list(text)
                if len(digits) < target_len:
                    digits += [''] * (target_len - len(digits))
                else:
                    digits = digits[:target_len]

                if len(confs) < target_len:
                    confs = confs + [0.5] * (target_len - len(confs))
                else:
                    confs = confs[:target_len]

                per_source_digits[t] = (digits, confs)

            # 逐字符挑选“置信度最高”的候选
            fused_digits = []
            fused_confs = []
            for i in range(target_len):
                best = None
                best_conf = -1.0
                best_src_rank = 999
                for t in candidates:
                    digits, confs = per_source_digits.get(t, ([], []))
                    d = digits[i] if i < len(digits) else ''
                    c = confs[i] if i < len(confs) else 0.5
                    # 如果该位为空字符，降权
                    if d == '':
                        c = min(c, 0.3)
                    # 并列时按 preferred_order 选择
                    rank = preferred_order.index(t) if t in preferred_order else 999
                    if (c > best_conf) or (abs(c - best_conf) < 1e-9 and rank < best_src_rank):
                        best = d
                        best_conf = c
                        best_src_rank = rank
                fused_digits.append(best if best is not None and best != '' else '0')
                fused_confs.append(float(best_conf))

            fused_text = ''.join(fused_digits)

            # 组 mixed 的 entry
            correct = (ground_truth == fused_text)
            cer = calculate_cer(ground_truth, fused_text) if ground_truth else 0.0

            entry = {
                'filename': fname,
                'image_path': img_path,
                'ground_truth': ground_truth,
                'recognized': fused_text,
                'content_type': 'digit',
                'correct': correct,
                'cer': cer,
                'digit_confidences': fused_confs,
                'confidence_based_result': fused_text,
                'confidence_based_correct': correct,
                'cer_confidence': cer,
            }
            mixed_entries.append(entry)

        else:
            # 英文：多数投票（忽略大小写）；若并列取最短
            votes = {}
            for t, r in candidates.items():
                text = (r.get('recognized') or '').strip().lower()
                if text:
                    votes[text] = votes.get(text, 0) + 1
            if votes:
                max_cnt = max(votes.values())
                tied = [w for w, c in votes.items() if c == max_cnt]
                fused_word = min(tied, key=len)
            else:
                # 如果都没识别到，就选优先顺序里的第一个非空
                fused_word = ''
                for t in preferred_order:
                    if t in candidates:
                        txt = (candidates[t].get('recognized') or '').strip().lower()
                        if txt:
                            fused_word = txt
                            break

            correct = (ground_truth.lower() == fused_word.lower())
            cer = calculate_cer(ground_truth.lower(), fused_word.lower()) if ground_truth else 0.0

            entry = {
                'filename': fname,
                'image_path': img_path,
                'ground_truth': ground_truth,
                'recognized': fused_word,
                'content_type': 'english',
                'correct': correct,
                'cer': cer,
            }
            mixed_entries.append(entry)

    return mixed_entries


def _calculate_accuracy_statistics(results: Dict[str, List]) -> Dict:
    """
    计算各种图片类型的准确率和CER统计
    
    Args:
        results: 识别结果字典
        
    Returns:
        准确率统计字典
    """
    print("\n" + "="*80)
    print("识别结果统计")
    print("="*80)
    
    accuracy_results = {}
    
    for img_type, result_list in results.items():
        if not result_list:
            accuracy_results[img_type] = {'correct': 0, 'total': 0, 'accuracy': 0, 'cer': 0}
            continue
        
        # 分别统计数字和英文的准确率和CER
        digit_results = [r for r in result_list if r['content_type'] == 'digit']
        english_results = [r for r in result_list if r['content_type'] == 'english']
        
        # 数字统计
        if digit_results:
            digit_stats = _calculate_stats(digit_results)
            accuracy_results[f"{img_type}_digit"] = digit_stats
            print(f"\n【{img_type} - 数字】")
            print(f"  准确率: {digit_stats['accuracy']:.2%} ({digit_stats['correct']}/{digit_stats['total']})")
            print(f"  CER:    {digit_stats['cer']:.4f}")
        
        # 英文统计
        if english_results:
            english_stats = _calculate_stats(english_results)
            accuracy_results[f"{img_type}_english"] = english_stats
            print(f"\n【{img_type} - 英文】")
            print(f"  准确率: {english_stats['accuracy']:.2%} ({english_stats['correct']}/{english_stats['total']})")
            print(f"  CER:    {english_stats['cer']:.4f}")
        
        # 总体统计
        total_stats = _calculate_stats(result_list)
        accuracy_results[img_type] = total_stats
        print(f"\n【{img_type} - 总体】")
        print(f"  准确率: {total_stats['accuracy']:.2%} ({total_stats['correct']}/{total_stats['total']})")
        print(f"  CER:    {total_stats['cer']:.4f}")
    
    print("="*80)
    return accuracy_results


def _calculate_stats(result_list: List[Dict]) -> Dict:
    """
    计算单个结果列表的统计信息
    
    Args:
        result_list: 结果列表
        
    Returns:
        统计字典
    """
    correct = sum(1 for r in result_list if r['correct'])
    total = len(result_list)
    accuracy = correct / total if total > 0 else 0
    cer = sum(r['cer'] for r in result_list) / total if total > 0 else 0
    
    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'cer': cer
    }


# =============================================================================
# CSV输出函数
# =============================================================================


def save_results_to_csv(results: Dict[str, List], output_path: str):
    """
    将识别结果保存到CSV文件，按ground_truth分组
    
    Args:
        results: 识别结果字典
        output_path: 输出CSV文件路径
    """
    # 按ground_truth分组整理数据
    grouped_results = {}
    
    for img_type, result_list in results.items():
        for result in result_list:
            ground_truth = result['ground_truth']
            if ground_truth not in grouped_results:
                grouped_results[ground_truth] = {}
            
            grouped_results[ground_truth][img_type] = {
                'filename': result['filename'],
                'image_path': result['image_path'],
                'recognized': result['recognized'],
                'correct': result['correct'],
                'content_type': result.get('content_type', 'digit'),
                'digit_confidences': result.get('digit_confidences', []),
                'confidence_based_result': result.get('confidence_based_result', ''),
                'confidence_based_correct': result.get('confidence_based_correct', False)
            }
    
    # 定义CSV字段
    fieldnames = [
        'ground_truth', 'content_type',
        'no_smoothing_result', 'no_smoothing_correct', 'no_smoothing_conf', 'no_smoothing_conf_result',
        'plain_result', 'plain_correct', 'plain_conf', 'plain_conf_result',
        'segmented_result', 'segmented_correct', 'segmented_conf', 'segmented_conf_result',
        'projection_result', 'projection_correct', 'projection_conf', 'projection_conf_result',
        'clean_projection_result', 'clean_projection_correct', 'clean_projection_conf', 'clean_projection_conf_result',
        'myself_result', 'myself_correct', 'myself_conf', 'myself_conf_result',
        'no_smoothing_file', 'plain_file', 'segmented_file',
        'projection_file', 'clean_projection_file', 'myself_file'
    ]
    
    # 写入CSV文件
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 按ground_truth排序输出
        for ground_truth in sorted(grouped_results.keys()):
            group_data = grouped_results[ground_truth]
            
            # 获取内容类型（假设同一个ground_truth的内容类型相同）
            content_type = 'digit'
            for img_type in ['no_smoothing', 'plain', 'segmented', 'projection', 'clean_projection', 'myself']:
                if img_type in group_data and 'content_type' in group_data[img_type]:
                    content_type = group_data[img_type]['content_type']
                    break
            
            row = {
                'ground_truth': ground_truth,
                'content_type': content_type
            }
            
            # 填充各种图片类型的结果
            for img_type in ['no_smoothing', 'plain', 'segmented', 'projection', 'clean_projection', 'myself']:
                if img_type in group_data:
                    row[f'{img_type}_result'] = group_data[img_type]['recognized']
                    row[f'{img_type}_correct'] = group_data[img_type]['correct']
                    row[f'{img_type}_file'] = group_data[img_type]['filename']
                    
                    # 添加置信度信息（仅数字识别）
                    confidences = group_data[img_type].get('digit_confidences', [])
                    if confidences:
                        conf_str = ', '.join([f"{c:.2f}" for c in confidences])
                        row[f'{img_type}_conf'] = conf_str
                        row[f'{img_type}_conf_result'] = group_data[img_type].get('confidence_based_result', '')
                    else:
                        row[f'{img_type}_conf'] = ''
                        row[f'{img_type}_conf_result'] = ''
                else:
                    row[f'{img_type}_result'] = ''
                    row[f'{img_type}_correct'] = ''
                    row[f'{img_type}_file'] = ''
                    row[f'{img_type}_conf'] = ''
                    row[f'{img_type}_conf_result'] = ''
            
            writer.writerow(row)
    
    print(f"详细结果已保存到: {output_path}")


# =============================================================================
# 主函数
# =============================================================================


def main():
    """
    主函数：执行批量文字识别（数字和英文单词）
    """
    # 配置参数
    RESULTS_DIR = "data/results"
    API_KEY = "sk-18258966b4494fd9a55f32b03f299bc8"
    
    # 参数检查
    if not API_KEY:
        print("错误: 请设置API密钥")
        return
    
    if not os.path.exists(RESULTS_DIR):
        print(f"错误: 结果目录不存在: {RESULTS_DIR}")
        return
    
    # 开始识别
    print("="*80)
    print("开始批量识别手写文字")
    print("="*80)
    
    results = batch_recognize_text(RESULTS_DIR, API_KEY)
    
    # 打印总结表格
    print("\n" + "="*80)
    print("最终统计总结")
    print("="*80)
    print(f"{'类型':<25} {'准确率':<15} {'CER':<10}")
    print("-"*80)
    
    for metric_type, stats in sorted(results['accuracy_summary'].items()):
        if stats['total'] > 0:
            acc_str = f"{stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})"
            cer_str = f"{stats['cer']:.4f}" if 'cer' in stats else "N/A"
            print(f"{metric_type:<25} {acc_str:<15} {cer_str:<10}")
    
    print("="*80)


if __name__ == "__main__":
    main()

