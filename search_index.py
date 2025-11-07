import numpy as np
import faiss
from typing import List, Tuple, Dict, Optional
import pickle
import os
import logging
from database import EmbeddingDatabase

class SimilaritySearchIndex:
    """基于FAISS的相似性搜索索引"""
    
    def __init__(self, db_path: str = "embeddings.db", index_path: str = "similarity_index.faiss"):
        self.db = EmbeddingDatabase(db_path)
        self.index_path = index_path
        self.mapping_path = index_path.replace('.faiss', '_mapping.pkl')
        self.index = None
        self.id_mapping = {}  # FAISS索引位置到句子ID的映射
        self.dimension = None
        
    def build_index(self, index_type: str = "flat") -> bool:
        """构建搜索索引"""
        try:
            logging.info("开始构建搜索索引...")
            
            # 获取所有嵌入向量
            embeddings_data = self.db.get_all_embeddings()
            
            if not embeddings_data:
                logging.warning("数据库中没有嵌入向量，无法构建索引")
                return False
            
            # 提取嵌入向量和ID映射
            sentence_ids = []
            embeddings = []
            
            for sentence_id, embedding in embeddings_data:
                sentence_ids.append(sentence_id)
                embeddings.append(embedding)
            
            embeddings_matrix = np.vstack(embeddings).astype('float32')
            self.dimension = embeddings_matrix.shape[1]
            
            logging.info(f"构建索引：{len(embeddings)} 个向量，维度 {self.dimension}")
            
            # 创建FAISS索引
            if index_type == "flat":
                self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
            elif index_type == "ivf":
                # IVF索引，适合大规模数据
                nlist = min(100, len(embeddings) // 10)  # 聚类中心数量
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                self.index.train(embeddings_matrix)
            else:
                raise ValueError(f"不支持的索引类型: {index_type}")
            
            # 添加向量到索引
            self.index.add(embeddings_matrix)
            
            # 创建ID映射
            self.id_mapping = {i: sentence_id for i, sentence_id in enumerate(sentence_ids)}
            
            # 保存索引和映射
            self.save_index()
            
            logging.info(f"索引构建完成，包含 {self.index.ntotal} 个向量")
            return True
            
        except Exception as e:
            logging.error(f"构建索引失败: {e}")
            return False
    
    def save_index(self):
        """保存索引到文件"""
        try:
            faiss.write_index(self.index, self.index_path)
            
            with open(self.mapping_path, 'wb') as f:
                pickle.dump({
                    'id_mapping': self.id_mapping,
                    'dimension': self.dimension
                }, f)
            
            logging.info(f"索引已保存到 {self.index_path}")
            
        except Exception as e:
            logging.error(f"保存索引失败: {e}")
    
    def load_index(self) -> bool:
        """从文件加载索引"""
        try:
            if not os.path.exists(self.index_path) or not os.path.exists(self.mapping_path):
                logging.warning(f"索引文件不存在: {self.index_path} 或 {self.mapping_path}")
                return False
            
            logging.info(f"正在加载索引文件: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            
            # 验证索引是否正确加载
            if self.index is None:
                raise ValueError("索引文件加载失败，返回None")
            
            logging.info(f"正在加载映射文件: {self.mapping_path}")
            with open(self.mapping_path, 'rb') as f:
                data = pickle.load(f)
                self.id_mapping = data['id_mapping']
                self.dimension = data['dimension']
            
            # 安全地访问ntotal属性
            try:
                vector_count = self.index.ntotal
                logging.info(f"索引加载完成，包含 {vector_count} 个向量")
            except AttributeError:
                logging.warning("无法获取索引向量数量，但索引已加载")
            
            return True
            
        except Exception as e:
            logging.error(f"加载索引失败: {e}")
            logging.error(f"索引文件路径: {self.index_path}")
            logging.error(f"映射文件路径: {self.mapping_path}")
            # 确保在加载失败时重置索引
            self.index = None
            self.id_mapping = {}
            self.dimension = None
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10, 
               threshold: float = 0.0) -> List[Dict]:
        """搜索最相似的句子"""
        if self.index is None:
            if not self.load_index():
                logging.error("索引未加载")
                return []
        
        try:
            # 确保查询向量格式正确
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            query_embedding = query_embedding.astype('float32')
            
            # 执行搜索
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue
                    
                if score < threshold:  # 过滤低分结果
                    continue
                
                # 获取句子ID
                sentence_id = self.id_mapping.get(idx)
                if sentence_id is None:
                    continue
                
                # 获取句子详细信息
                sentence_info = self.db.get_sentence_by_id(sentence_id)
                if sentence_info:
                    sentence_info['similarity_score'] = float(score)
                    sentence_info['rank'] = i + 1
                    results.append(sentence_info)
            
            return results
            
        except Exception as e:
            logging.error(f"搜索失败: {e}")
            return []
    
    def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        if self.index is None:
            if not self.load_index():
                return {}
        
        # 确保索引已正确加载
        if self.index is None:
            return {}
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': type(self.index).__name__,
            'mapping_size': len(self.id_mapping)
        }
    
    def rebuild_index(self, index_type: str = "flat") -> bool:
        """重建索引"""
        logging.info("重建搜索索引...")
        
        # 清理现有索引
        self.index = None
        self.id_mapping = {}
        
        # 删除旧的索引文件
        for path in [self.index_path, self.mapping_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # 重新构建
        return self.build_index(index_type)
    
    def add_vectors(self, sentence_ids: List[int], embeddings: np.ndarray):
        """向现有索引添加新向量"""
        if self.index is None:
            logging.error("索引未初始化")
            return False
        
        try:
            embeddings = embeddings.astype('float32')
            
            # 添加向量到索引
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            
            # 更新ID映射
            for i, sentence_id in enumerate(sentence_ids):
                self.id_mapping[start_idx + i] = sentence_id
            
            # 保存更新后的索引
            self.save_index()
            
            logging.info(f"成功添加 {len(sentence_ids)} 个向量到索引")
            return True
            
        except Exception as e:
            logging.error(f"添加向量失败: {e}")
            return False