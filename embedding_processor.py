import os
import re
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import logging
from database import EmbeddingDatabase

class SentenceEmbeddingProcessor:
    """句子嵌入批处理器"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "embeddings.db"):
        self.model_name = model_name
        self.model = None
        self.db = EmbeddingDatabase(db_path)
        self.load_model()
        
    def load_model(self):
        """加载句子嵌入模型"""
        try:
            logging.info(f"正在加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logging.info("模型加载完成")
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise
    
    def extract_sentences_from_file(self, file_path: str) -> List[str]:
        """从文件中提取句子，支持多种格式"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.md':
            return self.extract_sentences_from_markdown(file_path)
        elif file_extension == '.txt':
            return self.extract_sentences_from_txt(file_path)
        else:
            logging.warning(f"不支持的文件格式: {file_extension}")
            return []
    
    def extract_sentences_from_txt(self, file_path: str) -> List[str]:
        """从TXT文件中提取句子"""
        sentences = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 直接使用句子分割方法，不需要清理Markdown格式
            sentences = self.split_sentences(content)
                    
        except Exception as e:
            logging.error(f"读取TXT文件失败 {file_path}: {e}")
            
        return sentences
    
    def extract_sentences_from_markdown(self, file_path: str) -> List[str]:
        """从Markdown文件中提取句子"""
        sentences = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 移除Markdown格式标记
            content = self.clean_markdown(content)
            
            # 改进的句子分割，支持中英文标点符号
            sentences = self.split_sentences(content)
                    
        except Exception as e:
            logging.error(f"读取文件失败 {file_path}: {e}")
            
        return sentences
    
    def split_sentences(self, text: str) -> List[str]:
        """改进的句子分割方法，支持中英文"""
        sentences = []
        
        # 支持中英文句号、感叹号、问号的正则表达式
        # 包括：. ! ? 。 ！ ？
        sentence_pattern = r'[.!?。！？]+(?:\s+|$)'
        
        # 分割句子
        raw_sentences = re.split(sentence_pattern, text)
        
        for sentence in raw_sentences:
            sentence = sentence.strip()
            # 过滤条件：
            # 1. 长度大于5个字符（降低了门槛）
            # 2. 包含至少一个字母或中文字符
            # 3. 不是纯数字或符号
            if (len(sentence) > 5 and 
                re.search(r'[a-zA-Z\u4e00-\u9fff]', sentence) and
                not re.match(r'^[\d\s\-_=+*/#@$%^&()[\]{}|\\:;"\'<>,./]*$', sentence)):
                sentences.append(sentence)
        
        return sentences
    
    def clean_markdown(self, text: str) -> str:
        """清理Markdown格式"""
        # 移除标题标记
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # 移除链接
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # 移除图片
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        
        # 移除代码块
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', '', text)
        
        # 移除粗体和斜体标记
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)
        
        # 移除多余的空白字符
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def process_file(self, file_path: str, batch_size: int = 32) -> int:
        """处理单个文件，返回处理的句子数量"""
        filename = os.path.basename(file_path)
        logging.info(f"开始处理文件: {filename}")
        
        # 添加文档记录
        document_id = self.db.add_document(filename, file_path)
        
        # 提取句子（支持多种文件格式）
        sentences = self.extract_sentences_from_file(file_path)
        if not sentences:
            logging.warning(f"文件 {filename} 中未找到有效句子")
            return 0
        
        logging.info(f"从 {filename} 中提取到 {len(sentences)} 个句子")
        
        # 批量生成嵌入
        processed_count = 0
        for i in tqdm(range(0, len(sentences), batch_size), desc=f"处理 {filename}"):
            batch_sentences = sentences[i:i + batch_size]
            
            try:
                # 生成嵌入
                embeddings = self.model.encode(batch_sentences, convert_to_numpy=True)
                
                # 保存到数据库
                for j, (sentence, embedding) in enumerate(zip(batch_sentences, embeddings)):
                    sentence_index = i + j
                    self.db.add_sentence_embedding(
                        document_id, sentence, sentence_index, embedding
                    )
                    processed_count += 1
                    
            except Exception as e:
                logging.error(f"处理批次失败 {i}-{i+batch_size}: {e}")
                continue
        
        logging.info(f"文件 {filename} 处理完成，共处理 {processed_count} 个句子")
        return processed_count
    
    def process_directory(self, directory_path: str, file_patterns: List[str] = None, 
                         batch_size: int = 32) -> Dict[str, int]:
        """批量处理目录中的文件"""
        import glob
        
        if file_patterns is None:
            file_patterns = ["*.md", "*.txt"]  # 默认支持md和txt文件
        
        files = []
        for pattern in file_patterns:
            pattern_path = os.path.join(directory_path, pattern)
            files.extend(glob.glob(pattern_path, recursive=True))
        
        if not files:
            logging.warning(f"在目录 {directory_path} 中未找到匹配文件")
            return {}
        
        logging.info(f"找到 {len(files)} 个文件待处理")
        
        results = {}
        total_sentences = 0
        
        for file_path in files:
            try:
                count = self.process_file(file_path, batch_size)
                results[file_path] = count
                total_sentences += count
            except Exception as e:
                logging.error(f"处理文件失败 {file_path}: {e}")
                results[file_path] = 0
        
        logging.info(f"批量处理完成，总共处理 {total_sentences} 个句子")
        return results
    
    def get_embedding_dimension(self) -> int:
        """获取嵌入向量的维度"""
        if self.model is None:
            return 0
        
        # 使用一个测试句子获取维度
        test_embedding = self.model.encode(["test sentence"])
        return test_embedding.shape[1]
    
    def encode_query(self, query: str) -> np.ndarray:
        """对查询文本进行编码"""
        return self.model.encode([query])[0]