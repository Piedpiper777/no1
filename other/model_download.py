from modelscope import snapshot_download
import os

# 指定下载目录
model_save_dir = "/home/zhanggu/Project/tianchi/model"

# 确保目录存在
os.makedirs(model_save_dir, exist_ok=True)

# 下载text2vec-base-chinese模型到指定目录
model_dir = snapshot_download(
    'thomas/text2vec-base-chinese',
    cache_dir=model_save_dir,
    local_dir=os.path.join(model_save_dir, 'text2vec-base-chinese')
)

print(f"模型已下载到: {model_dir}")
