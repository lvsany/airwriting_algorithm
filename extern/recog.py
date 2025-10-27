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
from transformers import AutoModel, AutoTokenizer
import torch

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
        api_key = cfg('DASHSCOPE_API_KEY', '')
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


class LlamaDashScope(BaseProvider):
    """DashScope Llama multimodal model wrapper.

    Uses the same DashScope multimodal generation endpoint but targets
    Llama models (e.g. 'llama-4-maverick-17b-128e-instruct').
    Configure model with DASHSCOPE_LLAMA_MODEL env var or default to
    'llama-4-maverick-17b-128e-instruct'.
    """
    name = "llama"

    async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
        api_key = cfg('DASHSCOPE_API_KEY', '')
        if not api_key:
            return {'text': '', 'raw': '', 'error': 'Missing DASHSCOPE_API_KEY'}

        model = cfg('DASHSCOPE_LLAMA_MODEL', 'llama-4-maverick-17b-128e-instruct')
        url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
        prompt = get_prompt(image_type, content_type)

        payload = {
            'model': model,
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
            # parse body even on non-200 to extract structured error info
            try:
                j = resp.json()
            except Exception:
                j = {}

            if resp.status_code != 200:
                code = j.get('code') or j.get('error') or ''
                message = j.get('message') or ''
                request_id = j.get('request_id') or j.get('requestId') or resp.headers.get('x-request-id') or ''
                # Access denied often means the key does not have permission to call this model
                if 'AccessDenied' in str(code) or 'AccessDenied' in str(message):
                    # Optionally fallback to another model (e.g., qwen-vl-max)
                    if cfg('DASHSCOPE_LLAMA_FALLBACK', '1') in ('1', 'true', 'True'):
                        fallback_model = cfg('DASHSCOPE_LLAMA_FALLBACK_MODEL', 'qwen-vl-max')
                        payload['model'] = fallback_model
                        try:
                            resp2 = await self.client.post(url, headers=headers, json=payload, timeout=60)
                            resp2.raise_for_status()
                            j2 = resp2.json()
                            content = j2.get('output', {}).get('choices', [])[0].get('message', {}).get('content', '')
                            content_text = _normalize_content(content)
                            return self.parse_result(content_text, content_type)
                        except Exception as e2:
                            try:
                                err2 = resp2.content.decode('utf-8', errors='ignore')
                            except Exception:
                                err2 = str(e2)
                            return {'text': '', 'raw': '', 'error': f"llama/dashscope fallback error: {err2} (request_id:{request_id})"}
                    return {'text': '', 'raw': '', 'error': f"llama/dashscope AccessDenied: {message} (request_id:{request_id})"}

                # other non-200 errors: return truncated body for debugging
                try:
                    body = resp.content.decode('utf-8', errors='ignore')
                except Exception:
                    body = str(j)[:400]
                return {'text': '', 'raw': '', 'error': f"llama/dashscope HTTP {resp.status_code}: {body[:400]}"}

            # 200 OK path
            content = j.get('output', {}).get('choices', [])[0].get('message', {}).get('content', '')
            content_text = _normalize_content(content)
            return self.parse_result(content_text, content_type)

        except Exception as e:
            # best-effort to include response body if available
            err = ''
            if 'resp' in locals():
                try:
                    err = resp.content.decode('utf-8', errors='ignore')
                except Exception:
                    err = str(e)
            else:
                err = str(e)
            return {'text': '', 'raw': '', 'error': f'llama/dashscope exception: {err}'}


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
        api_key = cfg('ZHIPUAI_API_KEY', '')
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
        api_key = cfg('ARK_API_KEY', '')
        if not api_key:
            return {'text': '', 'raw': '', 'error': 'Missing ARK_API_KEY'}

        base = cfg('ARK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
        url = f'{base}/chat/completions'
        prompt = get_prompt(image_type, content_type)

        model_name = cfg('ARK_MODEL', '')  # 强烈建议配置 ep-xxxx
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
# class HFServerlessVLM(BaseProvider):
#     """
#     Hugging Face Serverless Inference API (VLM)
#     - 默认模型: HuggingFaceM4/idefics2-8b-chatty
#     - 需要环境变量: HF_TOKEN
#     - 可选: HF_MODEL, HF_NO_CACHE=1 以关闭缓存
#     """
#     name = "hf"

#     async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
#         hf_token = os.getenv("HF_TOKEN", "")
#         if not hf_token:
#             return {"text": "", "raw": "", "error": "Missing HF_TOKEN"}

#         model_id = os.getenv("HF_MODEL", "HuggingFaceM4/idefics2-8b-chatty")
#         url = f"https://api-inference.huggingface.co/models/{model_id}"

#         # 将图片以 Markdown data-URL 形式嵌入（Idefics2 支持）
#         # Prompt 里继续使用你已有的 get_prompt 来强约束“只返回 JSON/只返回单词”
#         task_prompt = get_prompt(image_type, content_type)
#         image_markdown = f"![](data:image/png;base64,{image_b64})"
#         prompt = f"{image_markdown}\n\n{task_prompt}"

#         headers = {
#             "Authorization": f"Bearer {hf_token}",
#             "Content-Type": "application/json",
#         }
#         if os.getenv("HF_NO_CACHE", ""):
#             headers["x-use-cache"] = "0"

#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "temperature": 0.0,
#                 "max_new_tokens": 200
#             }
#         }

#         # 503 冷启动重试
#         for attempt in range(3):
#             try:
#                 resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
#                 if resp.status_code == 503:
#                     # 模型正在加载
#                     try:
#                         info = resp.json()
#                         wait_s = float(info.get("estimated_time", 2.0))
#                     except Exception:
#                         wait_s = 2.0
#                     await asyncio.sleep(min(6.0, max(1.5, wait_s)))
#                     continue

#                 resp.raise_for_status()
#                 data = resp.json()

#                 # 常见返回形态 1：list[{"generated_text": "..."}]
#                 if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
#                     content_text = str(data[0]["generated_text"])
#                 # 形态 2：{"generated_text": "..."}
#                 elif isinstance(data, dict) and "generated_text" in data:
#                     content_text = str(data["generated_text"])
#                 # 兜底
#                 else:
#                     content_text = str(data)

#                 return self.parse_result(content_text, content_type)

#             except Exception as e:
#                 # 最后一轮仍失败则回传错误文本（带 body 便于排错）
#                 if attempt == 2:
#                     try:
#                         body = resp.content.decode("utf-8", errors="ignore")  # type: ignore
#                     except Exception:
#                         body = ""
#                     return {"text": "", "raw": "", "error": f"hf: {e}; {body[:300]}"}

#         return {"text": "", "raw": "", "error": "hf: retry_exceeded"}
class HFServerlessVLM(BaseProvider):
    """
    Hugging Face Inference Providers (OpenAI兼容路由)
    需要环境变量：
      HF_TOKEN   — 细粒度 token，须勾选“Make calls to the Inference Providers”
    可选环境变量：
      HF_MODEL   — 形如 'Qwen/Qwen3-VL-8B-Instruct:novita'
      HF_BASE_URL— 默认 'https://router.huggingface.co/v1'
    """
    name = "hf"

    async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
        token = cfg("HF_TOKEN", "")
        if not token:
            return {"text": "", "raw": "", "error": "hf: Missing HF_TOKEN"}
        model = cfg("HF_MODEL", "Qwen/Qwen3-VL-8B-Instruct:novita")
        base = cfg("HF_BASE_URL", "https://router.huggingface.co/v1")
        url = f"{base}/chat/completions"

        prompt = get_prompt(image_type, content_type)
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }],
            "temperature": 0.0,
            "max_tokens": 200
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8"
        }

        # Honor HF_NO_CACHE flag from embedded config (if set)
        if cfg("HF_NO_CACHE", ""):
            headers["x-use-cache"] = "0"

        # 轻量重试：模型冷启动或调度波动
        for attempt in range(3):
            try:
                resp = await self.client.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code in (502, 503, 504):
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if resp.status_code in (403, 404):
                    body = resp.content.decode("utf-8", errors="ignore")
                    return {"text": "", "raw": "", "error":
                            f"hf: HTTP {resp.status_code} for model={model}; {body[:250]}"}
                resp.raise_for_status()
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                content_text = _normalize_content(content)
                return self.parse_result(content_text, content_type)
            except Exception as e:
                if attempt == 2:
                    try:
                        body = resp.content.decode("utf-8", errors="ignore")  # type: ignore
                    except Exception:
                        body = ""
                    return {"text": "", "raw": "", "error": f"hf: {e}; {body[:300]}"}

        return {"text": "", "raw": "", "error": "hf: retry_exceeded"}
    

# ----------------------------- Gemini (google-genai SDK) -----------------------------
class GeminiGenAI(BaseProvider):
    """
    Google Gen AI SDK (Gemini) provider.
    需要在 halo_config 中配置：
      - GEMINI_API_KEY        (必填)
      - GEMINI_MODEL          (可选，默认 'gemini-2.5-flash')
      - GEMINI_JSON_MODE      (可选，'1' 启用 JSON 强约束；仅对数字任务开启最有效)
    文档：
      - Image understanding（传图/多图/文件）：https://ai.google.dev/gemini-api/docs/image-understanding
      - Generate content API（JSON/系统指令等）：https://ai.google.dev/api/generate-content
      - Python SDK： https://googleapis.github.io/python-genai/
    """
    name = "gemini"

    def __init__(self, client: httpx.AsyncClient):
        super().__init__(client)
        # 惰性导入，避免未安装库时影响其它 provider
        try:
            from google import genai
            from google.genai import types  # noqa: F401
            self._genai = genai
            self._types = types
        except Exception as e:
            self._import_err = e
            self._genai = None
            self._types = None

    async def recognize(self, image_b64: str, image_type: str, content_type: str) -> Dict:
        if self._genai is None or self._types is None:
            return {'text': '', 'raw': '', 'error': f'gemini: SDK import failed: {self._import_err}'}

        api_key = cfg('GEMINI_API_KEY', '')
        if not api_key:
            return {'text': '', 'raw': '', 'error': 'gemini: Missing GEMINI_API_KEY'}

        model = cfg('GEMINI_MODEL', 'gemini-2.5-flash')
        prompt = get_prompt(image_type, content_type)

        # 组装图片 part（SDK 推荐 from_bytes；单图即可直传。）
        try:
            img_bytes = base64.b64decode(image_b64)
        except Exception:
            return {'text': '', 'raw': '', 'error': 'gemini: invalid base64 image'}

        types = self._types
        image_part = types.Part.from_bytes(data=img_bytes, mime_type='image/png')

        # JSON Mode：对“digit”任务强约束输出为 application/json（避免 markdown/fence）
        response_cfg = None
        if content_type == 'digit' and cfg('GEMINI_JSON_MODE', '1') in ('1', 'true', 'True'):
            try:
                response_cfg = types.GenerateContentConfig(response_mime_type="application/json")
            except Exception:
                response_cfg = None  # 旧版 SDK 兜底

        client = self._genai.Client(api_key=api_key)

        # 【重要提示】官方建议：单张图 + 文本时，文本应当放在图片 part 之后（有助于视觉-文本对齐）:contentReference[oaicite:1]{index=1}
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[image_part, prompt],
                config=response_cfg  # None 时自动忽略
            )
        except Exception as e:
            return {'text': '', 'raw': '', 'error': f'gemini: request failed: {e}'}

        # SDK 响应：文本在 resp.text；必要时可读取结构化 candidates。这里沿用你的 parse 逻辑。
        try:
            content_text = getattr(resp, 'text', '') or ''
            return self.parse_result(content_text, content_type)
        except Exception as e:
            return {'text': '', 'raw': '', 'error': f'gemini: parse failed: {e}'}


# ----------------------------- 本地配置载入（来自 extern/halo_config.py） -----------------------------
try:
    import halo_config
except Exception:
    halo_config = None


def cfg(key: str, default: str = '') -> str:
    """Return configuration values from embedded halo_config.

    This intentionally avoids reading environment variables per user request.
    If the embedded config is missing or does not contain the key, returns
    the provided default.
    """
    if halo_config is None:
        return default
    return getattr(halo_config, key, default)


# ----------------------------- 批处理与并发 -----------------------------

def encode_image_file(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


async def recognize_with_all(image_path: str, image_type: str, content_type: str, client: httpx.AsyncClient) -> Dict[str, Dict]:
    b64 = encode_image_file(image_path)
    providers = [
        GeminiGenAI(client),
        LlamaDashScope(client),
        QwenDashScope(client),
        ZhipuGLM4V(client),
        DoubaoVision(client),
        HFServerlessVLM(client),
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
                for prov in ['gemini','llama','dashscope','zhipu','doubao','hf','deepseek_local']:
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

    # 检查 KEY (use embedded config instead of environment variables per user request)
    required = ['DASHSCOPE_API_KEY', 'ZHIPUAI_API_KEY', 'HUNYUAN_API_KEY', 'ARK_API_KEY']
    missing = [k for k in required if not cfg(k, '')]
    if missing:
        print(f"[WARN] 缺少内置配置：{', '.join(missing)}（对应厂商将返回错误信息，不影响其它模型运行）")

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
