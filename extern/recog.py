"""
A refreshed recognition module using QwenVL for handwriting images.

Features:
 - Detailed, type-specific prompts for 'clean' (cleaned/projection) and 'original' (raw) traces
 - Supports digit (with per-digit confidences as JSON) and english recognition
 - Batch runner that processes subfolders and writes CSV summary

Usage:
  python extern/recog_new.py --results_dir data/results --api_key <KEY>

"""
import os
import re
import json
import base64
import csv
import argparse
from typing import Dict, List, Tuple, Optional
import requests
from pathlib import Path

try:
    import yaml
    CONFIG = None
    cfg_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = None


def calculate_cer(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    len_ref = len(reference)
    len_hyp = len(hypothesis)
    d = [[0] * (len_hyp + 1) for _ in range(len_ref + 1)]
    for i in range(len_ref + 1):
        d[i][0] = i
    for j in range(len_hyp + 1):
        d[0][j] = j
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len_ref][len_hyp] / len_ref


def extract_ground_truth_from_filename(filename: str) -> Tuple[str, str]:
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    if len(parts) < 2:
        return "", "digit"
    label = parts[-1]
    if label.isdigit():
        return label, 'digit'
    if label.isalpha():
        return label.lower(), 'english'
    numbers = re.findall(r'\d+', label)
    if numbers:
        return max(numbers, key=len), 'digit'
    words = re.findall(r'[a-zA-Z]+', label)
    if words:
        return max(words, key=len).lower(), 'english'
    return "", 'digit'


def get_prompt(image_type: str, content_type: str = 'digit') -> str:
    """Return a detailed prompt tailored to the image_type and content_type.

    image_type: e.g. 'clean', 'original', 'plain', 'segmented', 'projection', 'myself'
    content_type: 'digit' or 'english'
    """
    # Digit prompts require strict JSON output with confidences per digit
    if content_type == 'digit':
        base_digit_json = (
            '图片中为连笔手写的四个数字轨迹。请严格以JSON格式返回，每个数字附带置信度（0-1）：\n'
            '{\n  "digits": [\n    {"digit": "1", "confidence": 0.95},\n    {"digit": "2", "confidence": 0.90},\n    {"digit": "3", "confidence": 0.80},\n    {"digit": "4", "confidence": 0.70}\n  ]\n}'
        )

        prompts = {
            'clean': (
                base_digit_json + '\n\n这是经过投影分割/清理的图片，轨迹干净，红色分界线标记了数字边界，两个红色分界线之间为一个字符区域。虚线标志是经过自动化处理后识别可能为连笔段的部分，请仔细考虑，如果不加入虚线部分，每个字符区域可以匹配是个数字就不加考虑，如果无法匹配是个数字就加入考虑虚线部分重新匹配' 
                '请优先根据红线将图分割为各个数字，再对每个区域识别数字并给出置信度。'
            ),
            'original': (
                base_digit_json + '\n\n这是原始未处理的手写轨迹图片，可能包含噪声和连接笔画。' 
                '请尽量识别出四个数字，若不确定请给出低置信度（接近0.0）。'
            ),
            # 'projection': (
            #     base_digit_json + '\n\n这是投影分割可视化图片，通常带有分界线。请依据分界线分割并逐片识别。'
            # ),
            'segmented': (
                base_digit_json + '\n\n这是分割结果图，包含分段与虚线标注。请识别每段的数字并给出置信度。'
            ),
            'plain': (
                base_digit_json + '\n\n这是平滑后的轨迹图，噪声较少。请识别四个数字并给出置信度。'
            ),
            'myself': (
                base_digit_json + '\n\n这是用户自定义的图片，请直接识别并严格按JSON格式返回。'
            )
        }
        return prompts.get(image_type, prompts['plain'])

    # English prompts: prefer plain text word only
    prompts_en = {
        'clean': '这是一张清理过的英文单词轨迹图，轨迹和分界清晰。请直接返回单词，不要多余文字。',
        'original': '这是原始英文手写轨迹，可能有噪声。请识别并返回单词，不要多余文字。',
        # 'projection': '这是投影分割图，红线标记字母边界。请依据边界识别并返回单词。',
        'segmented': '这是分割图，包含虚线/分段，识别实线部分为主并返回单词。',
        'plain': '这是平滑后的英文轨迹图，识别并返回单词。',
        'myself': '用户自定义轨迹，请识别并返回单词。'
    }
    return prompts_en.get(image_type, prompts_en['plain'])


class QwenVLRecognizerNew:
    def __init__(self, api_key: str, api_url: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.api_url = api_url or 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
        self.timeout = timeout
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def recognize(self, image_path: str, image_type: str = 'plain', content_type: str = 'digit') -> Dict:
        prompt = get_prompt(image_type, content_type)
        try:
            image_b64 = self._encode_image(image_path)
            data = {
                'model': 'qwen-vl-max',
                'input': {
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                { 'type': 'image', 'image': f'data:image/png;base64,{image_b64}' },
                                { 'type': 'text', 'text': prompt }
                            ]
                        }
                    ]
                },
                'parameters': {
                    'temperature': 0.0,
                    'max_tokens': 400
                }
            }
            resp = requests.post(self.api_url, headers=self.headers, json=data, timeout=self.timeout)
            resp.raise_for_status()
            j = resp.json()
            content = j.get('output', {}).get('choices', [])[0].get('message', {}).get('content', '')
            # Normalize content to string
            if isinstance(content, list):
                parts = [item.get('text', '') for item in content if isinstance(item, dict)]
                content_text = ''.join(parts)
            else:
                content_text = str(content)

            if content_type == 'english':
                words = re.findall(r'[a-zA-Z]+', content_text)
                word = words[0].lower() if words else ''
                return {'text': word, 'raw': content_text}

            # digits: try parse JSON first
            parsed = self._parse_digits_json(content_text)
            if parsed:
                return parsed

            # fallback: extract digits
            # Use heuristic extractor to avoid picking up confidence decimals
            text = self._extract_digits_from_text(content_text, target_len=4)
            return {'text': text, 'raw': content_text, 'digit_confidences': [], 'confidence_based_result': text}

        except requests.RequestException as e:
            return {'text': '', 'raw': '', 'error': str(e)}
        except Exception as e:
            return {'text': '', 'raw': '', 'error': str(e)}

    def _parse_digits_json(self, text: str) -> Optional[Dict]:
        # Try to find a JSON block containing 'digits'
        m = re.search(r'\{[\s\S]*?"digits"[\s\S]*?\}', text)
        if not m:
            return None
        js = m.group(0)
        try:
            parsed = json.loads(js)
            if 'digits' in parsed and isinstance(parsed['digits'], list):
                digits = []
                confs = []
                for it in parsed['digits']:
                    d = str(it.get('digit', ''))
                    c = float(it.get('confidence', 0.0))
                    digits.append(d)
                    confs.append(c)
                text = ''.join(digits)
                confidence_based = ''.join([d for d, c in zip(digits, confs) if c >= 0.5])
                return {'text': text, 'raw': text, 'digit_confidences': confs, 'confidence_based_result': confidence_based, 'parsed_json': parsed}
        except Exception:
            return None
        return None

    def _extract_digits_from_text(self, text: str, target_len: int = 4) -> str:
        """Heuristic extraction of a digit sequence from free-form model output.

        Strategies (in order):
        1. Find exact \bdddd\b (4-digit token)
        2. Find repeated '"digit"\s*:\s*"?d"?' patterns and join first target_len
        3. Find digits not adjacent to dots (avoid decimal confidences) and take the first target_len
        4. As a last resort, take the longest contiguous digit sequence and crop to target_len
        """
        # 1) exact token of length target_len
        m = re.search(rf"\b\d{{{target_len}}}\b", text)
        if m:
            return m.group(0)

        # 2) explicit digit fields like "digit": "1" or 'digit': '1'
        digit_fields = re.findall(r'digit"?\s*[:=]\s*"?(\d)"?', text, flags=re.IGNORECASE)
        if digit_fields and len(digit_fields) >= 1:
            # join up to target_len, pad with '0' if missing
            digits = digit_fields[:target_len]
            if len(digits) < target_len:
                digits += ['0'] * (target_len - len(digits))
            return ''.join(digits)

        # 3) digits not adjacent to dot (avoid numbers like 0.95)
        singles = re.findall(r'(?<![\d\.])(\d)(?![\d\.])', text)
        if len(singles) >= target_len:
            return ''.join(singles[:target_len])

        # 4) longest contiguous digit sequence
        runs = re.findall(r'\d+', text)
        if runs:
            best = max(runs, key=len)
            if len(best) >= target_len:
                return best[:target_len]
            # if none long enough, try to concatenate short runs until reach target_len
            concat = ''.join(runs)
            return concat[:target_len]

        return ''


def batch_recognize(results_dir: str, api_key: str, output_csv: str = 'results/recognition_new.csv') -> Dict:
    if not os.path.exists(results_dir):
        print(f'Results dir not found: {results_dir}')
        return {}

    recognizer = QwenVLRecognizerNew(api_key)

    types = ['clean', 'original', 'plain', 'segmented', 'projection', 'myself']
    results: Dict[str, List[Dict]] = {t: [] for t in types}

    for t in types:
        sub = os.path.join(results_dir, t)
        if not os.path.exists(sub):
            continue
        files = sorted([f for f in os.listdir(sub) if f.lower().endswith('.png')])
        if not files:
            continue
        print(f'Processing {t}: {len(files)} images')
        for fname in files:
            gt, ctype = extract_ground_truth_from_filename(fname)
            if not gt:
                continue
            p = os.path.join(sub, fname)
            res = recognizer.recognize(p, image_type=t, content_type=ctype)
            recognized = res.get('text', '')
            cer = calculate_cer(gt, recognized)
            entry = {'filename': fname, 'image_path': p, 'ground_truth': gt, 'recognized': recognized, 'content_type': ctype, 'correct': gt == recognized, 'cer': cer}
            if ctype == 'digit' and res.get('digit_confidences'):
                entry['digit_confidences'] = res['digit_confidences']
                entry['confidence_based_result'] = res.get('confidence_based_result', '')
            results[t].append(entry)
            status = '✓' if entry['correct'] else '✗'
            confinfo = ''
            if 'digit_confidences' in entry:
                confinfo = ' | conf=' + ','.join([f'{c:.2f}' for c in entry['digit_confidences']])
            print(f"  {fname}: GT={gt} -> {recognized} {status}{confinfo}")

    # Save CSV
    save_results_to_csv(results, output_csv)

    # Compute simple summary
    summary = {}
    for k, lst in results.items():
        total = len(lst)
        correct = sum(1 for r in lst if r['correct'])
        cer = sum(r['cer'] for r in lst) / total if total else 0.0
        summary[k] = {'total': total, 'correct': correct, 'accuracy': correct / total if total else 0.0, 'cer': cer}

    return {'detailed': results, 'summary': summary}


def save_results_to_csv(results: Dict[str, List[Dict]], output_path: str):
    fieldnames = ['type', 'filename', 'ground_truth', 'recognized', 'correct', 'cer', 'digit_confidences']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, lst in results.items():
            for r in lst:
                row = {k: r.get(k, '') for k in ['filename', 'ground_truth', 'recognized', 'correct', 'cer']}
                row['type'] = t
                row['digit_confidences'] = ','.join([f'{c:.2f}' for c in r.get('digit_confidences', [])]) if r.get('digit_confidences') else ''
                writer.writerow(row)
    print(f'Saved recognition CSV: {output_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', default='results')
    p.add_argument('--api_key', default='')
    p.add_argument('--out', default='results/recognition_new.csv')
    args = p.parse_args()

    api_key = "sk-18258966b4494fd9a55f32b03f299bc8"

    res = batch_recognize(args.results_dir, api_key, args.out)
    print('\nSummary:')
    for k, v in res['summary'].items():
        print(f"  {k}: {v['accuracy']:.2%} ({v['correct']}/{v['total']}) CER={v['cer']:.4f}")


if __name__ == '__main__':
    main()
