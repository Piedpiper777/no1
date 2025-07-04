import base64
import json
import os
from typing import List, Optional
import requests
from openai import OpenAI

# 配置阿里云通义千问 API
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 确保设置了环境变量
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_image(image_path):
    """将图片编码为 base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"图片编码失败: {image_path}, 错误: {e}")
        return None

def build_prompt(question, options, retrieved_texts, figure_files):
    """构建发送给大模型的提示词"""
    prompt = f"""请根据以下信息回答问题：

问题：{question}

选项：
"""
    for option in options:
        prompt += f"{option}\n"
    
    prompt += "\n参考文本：\n"
    for i, text in enumerate(retrieved_texts, 1):
        prompt += f"{i}. {text}\n"
    
    if figure_files:
        prompt += f"\n相关图表：{len(figure_files)} 张图片将一并提供\n"
    
    prompt += """
请仔细分析文本和图表信息，选择正确答案，并简要说明理由。
请直接回答选项字母（A/B/C/D）。
"""
    return prompt

def call_llm(prompt: str, figure_files: List[str] = None, model: str = "qwen-vl-plus") -> str:
    """
    调用多模态大模型（支持文本和图片）
    
    Args:
        prompt: 文本提示
        figure_files: 图片文件路径列表（可选）
        model: 使用的模型名称
    
    Returns:
        模型的回答
    """
    try:
        # 构建消息内容
        content = []
        
        # 添加文本内容
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 如果有图片，添加图片到消息中
        if figure_files:
            for fig_path in figure_files:
                base64_image = encode_image(fig_path)
                if base64_image:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"大模型调用失败: {e}")
        return f"错误: {str(e)}"

def call_llm_with_context(question, options, retrieved_texts, figure_files=None, model: str = "qwen-vl-plus") -> str:
    """
    调用多模态大模型（接受结构化输入）
    
    Args:
        question: 问题文本
        options: 选项列表
        retrieved_texts: 检索到的文本列表
        figure_files: 图片文件路径列表（可选）
        model: 使用的模型名称
    
    Returns:
        模型的回答
    """
    prompt = build_prompt(question, options, retrieved_texts, figure_files)
    return call_llm(prompt, figure_files, model)
