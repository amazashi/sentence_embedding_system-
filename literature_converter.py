#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文献文件转换脚本
将 knowledge_01/files/ 中的MD文件转换为标准格式，适合句子嵌入系统处理
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

class LiteratureConverter:
    def __init__(self, source_dir: str, target_dir: str):
        """
        初始化转换器
        
        Args:
            source_dir: 源文件夹路径
            target_dir: 目标文件夹路径
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # 创建目标文件夹
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除数学公式中的复杂符号（保留基本内容）
        text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
        text = re.sub(r'\$.*?\$', '', text)
        
        # 移除引用标记 [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # 移除作者邮箱
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # 移除DOI信息
        text = re.sub(r'DOI:\s*\S+', '', text)
        
        # 移除特殊符号和格式标记
        text = re.sub(r'[©®™]', '', text)
        text = re.sub(r'\*+', '', text)
        
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        提取句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        # 按句号、问号、感叹号分割句子
        sentences = re.split(r'[.!?]+', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 过滤太短的句子
                # 确保句子以句号结尾
                if not sentence.endswith('.'):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def process_markdown_content(self, content: str) -> Tuple[str, List[str]]:
        """
        处理Markdown内容
        
        Args:
            content: 原始Markdown内容
            
        Returns:
            (标题, 句子列表)
        """
        lines = content.split('\n')
        title = ""
        text_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 提取标题
            if line.startswith('# ') and not title:
                title = line[2:].strip()
                continue
            
            # 跳过二级标题和更低级标题（作为分段标记）
            if line.startswith('#'):
                continue
                
            # 跳过表格、代码块等
            if line.startswith('|') or line.startswith('```') or line.startswith('---'):
                continue
                
            # 收集正文内容
            if len(line) > 20:  # 过滤太短的行
                text_content.append(line)
        
        # 合并文本内容
        full_text = ' '.join(text_content)
        full_text = self.clean_text(full_text)
        
        # 提取句子
        sentences = self.extract_sentences(full_text)
        
        return title, sentences
    
    def convert_file(self, source_file: Path) -> bool:
        """
        转换单个文件
        
        Args:
            source_file: 源文件路径
            
        Returns:
            转换是否成功
        """
        try:
            # 读取源文件
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 处理内容
            title, sentences = self.process_markdown_content(content)
            
            if not title:
                title = source_file.stem
            
            if len(sentences) < 5:  # 如果句子太少，跳过
                print(f"跳过文件 {source_file.name}：句子数量太少 ({len(sentences)})")
                return False
            
            # 生成标准格式内容
            standard_content = self.generate_standard_format(title, sentences)
            
            # 保存到目标文件
            target_file = self.target_dir / f"{source_file.stem}_standard.md"
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(standard_content)
            
            print(f"转换完成: {source_file.name} -> {target_file.name} ({len(sentences)} 个句子)")
            return True
            
        except Exception as e:
            print(f"转换文件 {source_file.name} 时出错: {str(e)}")
            return False
    
    def generate_standard_format(self, title: str, sentences: List[str]) -> str:
        """
        生成标准格式内容
        
        Args:
            title: 文档标题
            sentences: 句子列表
            
        Returns:
            标准格式的Markdown内容
        """
        content = f"# {title}\n\n"
        content += "这是一个专门设计用于句子嵌入系统的标准Markdown文件。每个句子都以适当的标点符号结束，确保系统能够正确识别和提取。\n\n"
        
        # 将句子分组，每组大约10-15个句子
        sentence_groups = []
        current_group = []
        
        for sentence in sentences:
            current_group.append(sentence)
            if len(current_group) >= 12:  # 每组12个句子
                sentence_groups.append(current_group)
                current_group = []
        
        if current_group:  # 添加剩余的句子
            sentence_groups.append(current_group)
        
        # 生成分组内容
        for i, group in enumerate(sentence_groups, 1):
            content += f"## 第{i}部分\n\n"
            
            # 每3个句子一段
            for j in range(0, len(group), 3):
                paragraph = group[j:j+3]
                content += ' '.join(paragraph) + "\n\n"
        
        # 添加结论
        content += "## 结论\n\n"
        content += f"本文档包含了{len(sentences)}个完整的句子，涵盖了原文献的主要内容。"
        content += "每个句子都有明确的语义和适当的标点符号。"
        content += "这种格式最适合句子嵌入系统进行处理和分析。\n"
        
        return content
    
    def convert_all_files(self) -> Tuple[int, int]:
        """
        转换所有文件
        
        Returns:
            (成功转换的文件数, 总文件数)
        """
        md_files = list(self.source_dir.glob("*.md"))
        total_files = len(md_files)
        success_count = 0
        
        print(f"开始转换 {total_files} 个MD文件...")
        
        for md_file in md_files:
            if self.convert_file(md_file):
                success_count += 1
        
        print(f"\n转换完成！成功转换 {success_count}/{total_files} 个文件")
        print(f"转换后的文件保存在: {self.target_dir}")
        
        return success_count, total_files

def main():
    """主函数"""
    # 设置路径
    source_dir = r"E:\My\tex\knowledge_01\files"
    target_dir = r"E:\My\tex\sentence_embedding_system\init_files\literature_standard"
    
    # 创建转换器并执行转换
    converter = LiteratureConverter(source_dir, target_dir)
    success_count, total_count = converter.convert_all_files()
    
    print(f"\n转换统计:")
    print(f"- 源文件夹: {source_dir}")
    print(f"- 目标文件夹: {target_dir}")
    print(f"- 成功转换: {success_count} 个文件")
    print(f"- 转换率: {success_count/total_count*100:.1f}%")

if __name__ == "__main__":
    main()