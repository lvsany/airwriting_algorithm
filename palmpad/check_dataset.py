"""
列出 HuggingFace 数据集里的文件路径（不下载），用来确认 allow_patterns 写法。
"""
from huggingface_hub import list_repo_files

files = list(list_repo_files("Teburile/Palmpad_Dataset", repo_type="dataset"))
print(f"Total files: {len(files)}")
for f in files[:40]:
    print(f)
