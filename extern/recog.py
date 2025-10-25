#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多模型并发手写识别（国内 & 免费/有免费额度）：
 - 支持：阿里 DashScope(Qwen-VL)、智谱 GLM-4V-Flash、腾讯混元 hunyuan-vision、字节 Doubao Vision
 - 数字：严格 JSON（每位 digit + confidence），英文：返回单词
 - 批处理：遍历 results_dir/{clean, original, plain, segmented, projection, myself}/*.png
 - 输出：逐模型 CER/准确率统计 + CSV 汇总（含每张图片各模型结果）

用法：
  python recog_multimodel.py --results_dir data/results --out results/recognition_multimodel.csv
环境变量：
  DASHSCOPE_API_KEY, ZHIPUAI_API_KEY, HUNYUAN_API_KEY, ARK_API_KEY
"""

import os
import re
import json
import base64
import csv
import argparse
import asyncio
from typing import Dict, List, Tuple, Optional

import httpx


# ----------------------------- 基础工具 -----------------------------

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
    image_type = (image_type or '').lower().strip()

    if content_type == 'digit':
        base_digit_json = (
            '图片中为连笔手写的四个数字轨迹。请严格以JSON格式返回，每个数字附带置信度（0-1）：\n'
            '{\n  "digits": [\n    {"digit": "1", "confidence": 0.95},\n'
            '    {"digit": "2", "confidence": 0.90},\n'
            '    {"digit": "3", "confidence": 0.80},\n'
            '    {"digit": "4", "confidence": 0.70}\n  ]\n}'
        )
        prompts = {
            'clean': (
                base_digit_json
                + '\n\n【输出要求】严格只返回一个 JSON 对象（不要任何解释、反引号或额外字符）。'
                  '确保 "digits" 恰好 4 个元素，顺序为从左到右；"digit" 取 "0"-"9" 单字符；'
                  '"confidence" 为 [0,1] 的小数。若很不确定，仍给出最可能数字，但将置信度设为接近 0.0。\n'
                  '【图像说明（干净版）】经过投影/清理，红线标记字符边界；虚线是可能的“连笔段”。\n'
                  '【判定流程】1) 先按红线切成4区；2) 各区识别并打分；3) 若某区不成形再谨慎并入相邻虚线段重试；'
                  '4) “近似直线、低曲率、整体斜向上”的段优先视作跨字符连笔（不并入）；5) 仅输出 JSON。'
            ),
            'original': (
                base_digit_json
                + '\n\n【输出要求】严格只返回一个 JSON 对象；"digits" 恰好4个，顺序自左向右；'
                  '"digit" 为 "0"-"9"；"confidence"∈[0,1]，不确定给低置信度。\n'
                  '【策略】尝试多种合理分割，选总体置信度最高且形态最合理的一种；仅输出 JSON。'
            ),
            'projection': base_digit_json + '\n\n这是投影分割可视化图片，请依据分界线分割并逐片识别，仅输出 JSON。',
            'segmented': base_digit_json + '\n\n这是分割结果图，包含虚线标注；优先识别实线主笔画，仅输出 JSON。',
            'plain': base_digit_json + '\n\n这是平滑后的轨迹图，识别四个数字并给出置信度，仅输出 JSON。',
            'myself': base_digit_json + '\n\n用户自定义图片，请直接识别并严格按 JSON 返回。'
        }
        return prompts.get(image_type, prompts['plain'])

    prompts_en = {
        'clean': '这是一张清理过的英文单词轨迹图，轨迹和分界清晰。请直接返回单词，不要多余文字。',
        'original': '这是原始英文手写轨迹，可能有噪声。请识别并返回单词，不要多余文字。',
        'projection': '这是投影分割图，红线标记字母边界。请依据边界识别并返回单词。',
        'segmented': '这是分割图，包含虚线/分段，识别实线部分为主并返回单词。',
        'plain': '这是平滑后的英文轨迹图，识别并返回单词。',
        'myself': '用户自定义轨迹，请识别并返回单词。'
    }
    return prompts_en.get(image_type, prompts_en['plain'])


def _parse_digits_json(text: str) -> Optional[Dict]:
    m = re.search(r'\{[\s\S]*?"digits"[\s\S]*?\}', text)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        if isinstance(parsed.get('digits'), list):
            digits, confs = [], []
            for it in parsed['digits']:
                d = str(it.get('digit', ''))
                c = float(it.get('confidence', 0.0))
                digits.append(d)
                confs.append(c)
            concat = ''.join(digits)
            conf_based = ''.join([d for d, c in zip(digits, confs) if c >= 0.5])
            return {'text': concat, 'digit_confidences': confs, 'confidence_based_result': conf_based, 'raw': m.group(0)}
    except Exception:
        return None
    return None


def _extract_digits_from_text(text: str, target_len: int = 4) -> str:
    m = re.search(rf"\b\d{{{target_len}}}\b", text)
    if m:
        return m.group(0)
    digit_fields = re.findall(r'digit"?\s*[:=]\s*"?(\d)"?', text, flags=re.IGNORECASE)
    if digit_fields:
        ds = (digit_fields + ['0'] * target_len)[:target_len]
        return ''.join(ds)
    singles = re.findall(r'(?<![\d\.])(\d)(?![\d\.])', text)
    if len(singles) >= target_len:
        return ''.join(singles[:target_len])
    runs = re.findall(r'\d+', text)
    if runs:
        best = max(runs, key=len)
        if len(best) >= target_len:
            return best[:target_len]
        return (''.join(runs))[:target_len]
    return ''


def _normalize_content(content) -> str:
    if isinstance(content, list):
        # OpenAI 兼容：content 可能是 [{type:text|image_url|input_image,...}, ...]
        parts = []
        for it in content:
            if isinstance(it, dict):
                if 'text' in it:
                    parts.append(str(it['text']))
                elif 'image_url' in it and isinstance(it['image_url'], dict):
                    parts.append(it['image_url'].get('url', ''))
        return '\n'.join([p for p in parts if p])
    return str(content or '')


# ----------------------------- 提供商客户端 -----------------------------

class BaseProvider:
    name = "base"

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
        raise NotImplementedError

    @staticmethod
    def make_text_result(text: str) -> Dict:
        return {'text': text, 'raw': text}

    @staticmethod
    def parse_result(content_text: str, content_type: str) -> Dict:
        if content_type == 'english':
            words = re.findall(r'[a-zA-Z]+', content_text)
            word = words[0].lower() if words else ''
            return {'text': word, 'raw': content_text}
        # digit
        parsed = _parse_digits_json(content_text)
        if parsed:
            return parsed
        text = _extract_digits_from_text(content_text, target_len=4)
        return {'text': text, 'raw': content_text}


class QwenDashScope(BaseProvider):
    name = "dashscope"

    async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
        api_key = os.getenv('DASHSCOPE_API_KEY', '')
        if not api_key:
            return {'text': '', 'raw': '', 'error': 'Missing DASHSCOPE_API_KEY'}
        url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
        prompt = get_prompt(image_type, content_type)
        payload = {
            'model': 'qwen-vl-max',
            'input': {
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': f'data:image/png;base64,{image_b64}'},
                        {'type': 'text', 'text': prompt}
                    ]
                }]
            },
            'parameters': {'temperature': 0.0, 'max_tokens': 400}
        }
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        try:
            resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            j = resp.json()
            content = j.get('output', {}).get('choices', [])[0].get('message', {}).get('content', '')
            content_text = _normalize_content(content)
            return self.parse_result(content_text, content_type)
        except Exception as e:
            return {'text': '', 'raw': '', 'error': f'dashscope: {e}'}


# class ZhipuGLM4V(BaseProvider):
#     name = "zhipu"

#     async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
#         api_key = os.getenv('ZHIPUAI_API_KEY', '')
#         if not api_key:
#             return {'text': '', 'raw': '', 'error': 'Missing ZHIPUAI_API_KEY'}
#         url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
#         prompt = get_prompt(image_type, content_type)
#         payload = {
#             'model': 'glm-4v-flash',  # 官方标识（免费多模态）：
#             # 参考文档列出：glm-4v-flash / glm-4v-plus-0111
#             'messages': [{
#                 'role': 'user',
#                 'content': [
#                     {'type': 'text', 'text': prompt},
#                     {'type': 'input_image', 'image_url': {'url': f'data:image/png;base64,{image_b64}' }}
#                 ]
#             }],
#             'temperature': 0.0,
#             'max_tokens': 400
#         }
#         headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
#         try:
#             resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
#             resp.raise_for_status()
#             j = resp.json()
#             content = j.get('choices', [])[0].get('message', {}).get('content', '')
#             content_text = _normalize_content(content)
#             return self.parse_result(content_text, content_type)
#         except Exception as e:
#             return {'text': '', 'raw': '', 'error': f'zhipu: {e}'}
class ZhipuGLM4V(BaseProvider):
    name = "zhipu"

    async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
        api_key = os.getenv('ZHIPUAI_API_KEY', '')
        if not api_key:
            return {'text': '', 'raw': '', 'error': 'Missing ZHIPUAI_API_KEY'}
        url = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
        prompt = get_prompt(image_type, content_type)
        payload = {
            'model': 'glm-4v-flash',
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/png;base64,{image_b64}'}}  # ← 关键：对象 + url 字段
                ]
            }],
            'temperature': 0.0,
            'max_tokens': 400
        }
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json; charset=utf-8'}
        try:
            resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            j = resp.json()
            content = j.get('choices', [])[0].get('message', {}).get('content', '')
            content_text = _normalize_content(content)
            return self.parse_result(content_text, content_type)
        except Exception as e:
            try:
                err = resp.content.decode('utf-8', errors='ignore')
            except:
                err = str(e)
            return {'text': '', 'raw': '', 'error': f'zhipu: {err}'}

# class HunyuanVision(BaseProvider):
#     name = "hunyuan"

#     async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
#         api_key = os.getenv('HUNYUAN_API_KEY', '')
#         if not api_key:
#             return {'text': '', 'raw': '', 'error': 'Missing HUNYUAN_API_KEY'}
#         base = 'https://api.hunyuan.cloud.tencent.com/v1'
#         url = f'{base}/chat/completions'
#         prompt = get_prompt(image_type, content_type)
#         payload = {
#             'model': 'hunyuan-vision',
#             'messages': [{
#                 'role': 'user',
#                 'content': [
#                     {'type': 'text', 'text': prompt},
#                     {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}}
#                 ]
#             }],
#             'temperature': 0.0,
#             'max_tokens': 400
#         }
#         headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
#         try:
#             resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
#             resp.raise_for_status()
#             j = resp.json()
#             content = j.get('choices', [])[0].get('message', {}).get('content', '')
#             content_text = _normalize_content(content)
#             return self.parse_result(content_text, content_type)
#         except Exception as e:
#             return {'text': '', 'raw': '', 'error': f'hunyuan: {e}'}
class HunyuanVision(BaseProvider):
    name = "hunyuan"

    async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
        api_key = os.getenv('HUNYUAN_API_KEY', '')
        if not api_key:
            return {'text': '', 'raw': '', 'error': 'Missing HUNYUAN_API_KEY'}

        # 二选一：未设置则默认 OpenAI 兼容路径
        base = os.getenv('HUNYUAN_BASE_URL', 'https://hunyuan.cloud.tencent.com/openai/v1')
        url = f'{base}/chat/completions'
        prompt = get_prompt(image_type, content_type)

        payload = {
            'model': os.getenv('HUNYUAN_MODEL', 'hunyuan-vision'),
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}}
                ]
            }],
            'temperature': 0.0,
            'max_tokens': 400
        }
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json; charset=utf-8'}

        try:
            resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            j = resp.json()
            content = j.get('choices', [])[0].get('message', {}).get('content', '')
            content_text = _normalize_content(content)
            return self.parse_result(content_text, content_type)
        except Exception as e:
            # 用 bytes 解码，避免 ascii 报错
            try:
                err = resp.content.decode('utf-8', errors='ignore')
            except:
                err = str(e)
            return {'text': '', 'raw': '', 'error': f'hunyuan: {err}'}


# class DoubaoVision(BaseProvider):
#     name = "doubao"

#     async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
#         api_key = os.getenv('ARK_API_KEY', '')
#         if not api_key:
#             return {'text': '', 'raw': '', 'error': 'Missing ARK_API_KEY'}
#         # Chat API（北京地域）
#         base = 'https://ark.cn-beijing.volces.com/api/v3'
#         url = f'{base}/chat/completions'
#         prompt = get_prompt(image_type, content_type)
#         # 常见视觉模型示例：'doubao-1-5-vision-pro-32k'（不同租户可见的具体型号可能略有不同）
#         payload = {
#             'model': 'doubao-1-5-vision-pro-32k',
#             'messages': [{
#                 'role': 'user',
#                 'content': [
#                     {'type': 'text', 'text': prompt},
#                     {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}}
#                 ]
#             }],
#             'temperature': 0.0,
#             'max_tokens': 400
#         }
#         headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
#         try:
#             resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
#             resp.raise_for_status()
#             j = resp.json()
#             # OpenAI 兼容：content 可能是 str 或 list
#             content = j.get('choices', [])[0].get('message', {}).get('content', '')
#             content_text = _normalize_content(content)
#             return self.parse_result(content_text, content_type)
#         except Exception as e:
#             return {'text': '', 'raw': '', 'error': f'doubao: {e}'}
class DoubaoVision(BaseProvider):
    name = "doubao"

    async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
        api_key = os.getenv('ARK_API_KEY', '')
        if not api_key:
            return {'text': '', 'raw': '', 'error': 'Missing ARK_API_KEY'}

        base = os.getenv('ARK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
        url = f'{base}/chat/completions'
        prompt = get_prompt(image_type, content_type)

        model_name = os.getenv('ARK_MODEL', '')  # 强烈建议配置 ep-xxxx
        if not model_name or not model_name.startswith('ep-'):
            return {'text': '', 'raw': '', 'error': 'doubao: 请在环境变量 ARK_MODEL 配置已发布的 endpoint（形如 ep-xxxxxxxx）'}

        payload = {
            'model': model_name,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{image_b64}'}}
                ]
            }],
            'temperature': 0.0,
            'max_tokens': 400
        }
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json; charset=utf-8'}

        try:
            resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            j = resp.json()
            content = j.get('choices', [])[0].get('message', {}).get('content', '')
            content_text = _normalize_content(content)
            return self.parse_result(content_text, content_type)
        except Exception as e:
            try:
                err = resp.content.decode('utf-8', errors='ignore')
            except:
                err = str(e)
            return {'text': '', 'raw': '', 'error': f'doubao: {err}'}


# ----------------------------- 批处理与并发 -----------------------------

def encode_image_file(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


async def recognize_with_all(image_path: str, image_type: str, content_type: str, client: httpx.AsyncClient) -> Dict[str, Dict]:
    b64 = encode_image_file(image_path)
    providers = [
        QwenDashScope(client),
        ZhipuGLM4V(client),
        HunyuanVision(client),
        DoubaoVision(client)
    ]
    tasks = [p.recognize(b64, image_type, content_type) for p in providers]
    results = await asyncio.gather(*tasks)
    return {p.name: r for p, r in zip(providers, results)}


def save_results_to_csv(rows: List[Dict], output_path: str):
    fieldnames = [
        'type', 'filename', 'content_type',
        'provider', 'ground_truth', 'recognized',
        'correct', 'cer', 'digit_confidences'
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[CSV] Saved: {output_path}')


async def batch_recognize(results_dir: str, output_csv: str, types: list) -> Dict:
    all_rows: List[Dict] = []

    per_provider_stats = {}  # {provider: {'total':.., 'correct':.., 'cer_sum':..}}
    per_type_stats = {}  # {type: {provider: stats}}

    async with httpx.AsyncClient(timeout=60) as client:
        for t in types:
            sub = os.path.join(results_dir, t)
            if not os.path.exists(sub):
                continue
            files = sorted([f for f in os.listdir(sub) if f.lower().endswith('.png')])
            if not files:
                continue
            print(f'[{t}] {len(files)} images')
            for fname in files:
                gt, ctype = extract_ground_truth_from_filename(fname)
                if not gt:
                    continue
                p = os.path.join(sub, fname)
                model_outputs = await recognize_with_all(p, t, ctype, client)

                for provider, res in model_outputs.items():
                    recognized = res.get('text', '') or ''
                    cer = calculate_cer(gt, recognized)
                    row = {
                        'type': t,
                        'filename': fname,
                        'content_type': ctype,
                        'provider': provider,
                        'ground_truth': gt,
                        'recognized': recognized,
                        'correct': (gt == recognized),
                        'cer': f'{cer:.6f}',
                        'digit_confidences': ','.join([f'{c:.2f}' for c in res.get('digit_confidences', [])]) if res.get('digit_confidences') else ''
                    }
                    all_rows.append(row)

                    # update overall provider stats
                    st = per_provider_stats.setdefault(provider, {'total': 0, 'correct': 0, 'cer_sum': 0.0})
                    st['total'] += 1
                    st['cer_sum'] += cer
                    if row['correct']:
                        st['correct'] += 1

                    # update per-type provider stats
                    tstats = per_type_stats.setdefault(t, {})
                    pst = tstats.setdefault(provider, {'total': 0, 'correct': 0, 'cer_sum': 0.0})
                    pst['total'] += 1
                    pst['cer_sum'] += cer
                    if row['correct']:
                        pst['correct'] += 1

                parts = []
                for prov in ['dashscope','zhipu','hunyuan','doubao']:
                    if prov in model_outputs:
                        r = model_outputs[prov]
                        if r.get('text'):
                            parts.append(f"{prov}:{r['text']}")
                        else:
                            parts.append(f"{prov}:ERR")
                            if r.get('error'):
                                parts.append(f"[{prov}-err:{r['error'][:160]}]")
                pretty = ' | '.join(parts)
                print(f"  {fname}: GT={gt} -> {pretty}")


    save_results_to_csv(all_rows, output_csv)

    # 汇总 overall
    summary = {}
    for prov, st in per_provider_stats.items():
        total = st['total'] or 1
        acc = st['correct'] / st['total'] if st['total'] else 0.0
        cer_avg = st['cer_sum'] / total
        summary[prov] = {'total': st['total'], 'correct': st['correct'], 'accuracy': acc, 'cer': cer_avg}

    # 汇总 per type
    per_type_summary = {}
    for typ, tstats in per_type_stats.items():
        s = {}
        for prov, st in tstats.items():
            total = st['total'] or 1
            acc = st['correct'] / st['total'] if st['total'] else 0.0
            cer_avg = st['cer_sum'] / total
            s[prov] = {'total': st['total'], 'correct': st['correct'], 'accuracy': acc, 'cer': cer_avg}
        per_type_summary[typ] = s

    return {'summary': summary, 'per_type_summary': per_type_summary, 'rows': all_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results', help='根目录包含 clean/original/plain/segmented/projection/myself 子目录')
    ap.add_argument('--out', default='results/recognition_multimodel.csv')
    ap.add_argument('--types', default='clean,original,plain,segmented,projection,myself',
                    help='要识别的子目录，逗号分隔，例如 --types clean,original')
    args = ap.parse_args()

    # 检查 KEY
    missing = [k for k in ['DASHSCOPE_API_KEY','ZHIPUAI_API_KEY','HUNYUAN_API_KEY','ARK_API_KEY'] if not os.getenv(k)]
    if missing:
        print(f"[WARN] 缺少环境变量：{', '.join(missing)}（对应厂商将返回错误信息，不影响其它模型运行）")

    types_list = [t.strip() for t in args.types.split(',') if t.strip()]
    res = asyncio.run(batch_recognize(args.results_dir, args.out, types_list))

    # Print per-type summaries (e.g., clean, original)
    per_type = res.get('per_type_summary', {})
    for typ in types_list:
        if typ not in per_type:
            continue
        print(f"\n=== Summary by provider ({typ}) ===")
        for prov, v in per_type[typ].items():
            print(f"  {prov:9s}  acc={v['accuracy']:.2%}  ({v['correct']}/{v['total']})  CER={v['cer']:.4f}")

    # Also print overall summary for convenience
    print('\n=== Summary by provider (overall) ===')
    for prov, v in res['summary'].items():
        print(f"  {prov:9s}  acc={v['accuracy']:.2%}  ({v['correct']}/{v['total']})  CER={v['cer']:.4f}")


if __name__ == '__main__':
    main()
