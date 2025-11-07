import sqlite3
import numpy as np
import json
from typing import List, Tuple, Optional
import logging

class EmbeddingDatabase:
    """管理句子嵌入的SQLite数据库"""
    
    def __init__(self, db_path: str = "embeddings.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建文档表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_path)
                )
            ''')
            
            # 创建句子表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    sentence_text TEXT NOT NULL,
                    sentence_index INTEGER,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            # 创建索引以提高查询性能
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_id ON sentences(document_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentence_index ON sentences(sentence_index)')
            
            conn.commit()
            logging.info("数据库初始化完成")
    
    def add_document(self, filename: str, file_path: str) -> int:
        """添加文档记录，返回文档ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    'INSERT INTO documents (filename, file_path) VALUES (?, ?)',
                    (filename, file_path)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # 文档已存在，返回现有ID
                cursor.execute('SELECT id FROM documents WHERE file_path = ?', (file_path,))
                return cursor.fetchone()[0]
    
    def add_sentence_embedding(self, document_id: int, sentence_text: str, 
                             sentence_index: int, embedding: np.ndarray):
        """添加句子嵌入"""
        embedding_blob = embedding.tobytes()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sentences (document_id, sentence_text, sentence_index, embedding)
                VALUES (?, ?, ?, ?)
            ''', (document_id, sentence_text, sentence_index, embedding_blob))
            conn.commit()
    
    def get_all_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """获取所有嵌入向量"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, embedding FROM sentences WHERE embedding IS NOT NULL')
            results = []
            for row in cursor.fetchall():
                sentence_id, embedding_blob = row
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                results.append((sentence_id, embedding))
            return results
    
    def get_sentence_by_id(self, sentence_id: int) -> Optional[dict]:
        """根据ID获取句子信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.id, s.sentence_text, s.sentence_index, d.filename, d.file_path
                FROM sentences s
                JOIN documents d ON s.document_id = d.id
                WHERE s.id = ?
            ''', (sentence_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'text': row[1],
                    'index': row[2],
                    'filename': row[3],
                    'file_path': row[4]
                }
            return None
    
    def get_document_sentences(self, document_id: int) -> List[dict]:
        """获取文档的所有句子"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, sentence_text, sentence_index
                FROM sentences
                WHERE document_id = ?
                ORDER BY sentence_index
            ''', (document_id,))
            
            return [
                {'id': row[0], 'text': row[1], 'index': row[2]}
                for row in cursor.fetchall()
            ]
    
    def clear_database(self):
        """清空数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sentences')
            cursor.execute('DELETE FROM documents')
            conn.commit()
            logging.info("数据库已清空")
    
    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM documents')
            doc_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM sentences')
            sentence_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM sentences WHERE embedding IS NOT NULL')
            embedding_count = cursor.fetchone()[0]
            
            return {
                'documents': doc_count,
                'sentences': sentence_count,
                'embeddings': embedding_count
            }
    
    def get_all_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """获取所有嵌入向量"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.id, s.embedding 
                FROM sentences s 
                WHERE s.embedding IS NOT NULL
                ORDER BY s.id
            ''')
            
            results = []
            for row in cursor.fetchall():
                sentence_id, embedding_blob = row
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                results.append((sentence_id, embedding))
            
            return results
    
    def get_sentence_by_index(self, index: int) -> Optional[Tuple]:
        """根据索引获取句子信息（索引从0开始）"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.id, s.sentence_index, s.sentence_text, s.document_id, d.filename
                FROM sentences s
                JOIN documents d ON s.document_id = d.id
                WHERE s.embedding IS NOT NULL
                ORDER BY s.id
                LIMIT 1 OFFSET ?
            ''', (index,))
            
            return cursor.fetchone()