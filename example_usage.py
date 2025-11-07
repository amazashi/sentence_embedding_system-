#!/usr/bin/env python3
"""
句子嵌入系统使用示例
演示如何在Python代码中直接使用各个模块
"""

import os
import sys
from typing import List, Dict

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding_processor import SentenceEmbeddingProcessor
from search_index import SimilaritySearchIndex
from database import EmbeddingDatabase

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 1. 初始化处理器
    processor = SentenceEmbeddingProcessor(
        model_name="all-MiniLM-L6-v2",
        db_path="example_embeddings.db"
    )
    
    # 2. 处理示例文本
    sample_sentences = [
        "机器学习是人工智能的一个重要分支。",
        "深度学习使用神经网络来模拟人脑的工作方式。",
        "自然语言处理帮助计算机理解人类语言。",
        "计算机视觉让机器能够识别和理解图像。",
        "强化学习通过奖励机制来训练智能体。"
    ]
    
    # 3. 添加文档到数据库
    db = EmbeddingDatabase("example_embeddings.db")
    doc_id = db.add_document("example_doc.txt", "example_path")
    
    # 4. 生成并存储嵌入
    print("正在生成嵌入向量...")
    embeddings = processor.model.encode(sample_sentences)
    
    for i, (sentence, embedding) in enumerate(zip(sample_sentences, embeddings)):
        db.add_sentence_embedding(doc_id, sentence, i, embedding)
    
    print(f"已处理 {len(sample_sentences)} 个句子")
    
    # 5. 构建搜索索引
    print("正在构建搜索索引...")
    search_index = SimilaritySearchIndex(
        db_path="example_embeddings.db",
        index_path="example_index.faiss"
    )
    search_index.build_index()
    
    # 6. 执行搜索
    print("正在搜索相似句子...")
    query = "人工智能和神经网络"
    query_embedding = processor.encode_query(query)
    results = search_index.search(query_embedding, k=3)
    
    print(f"\n查询: '{query}'")
    print("搜索结果:")
    for result in results:
        print(f"  相似度: {result['similarity_score']:.4f}")
        print(f"  句子: {result['text']}")
        print()

def example_batch_processing():
    """批量处理示例"""
    print("=== 批量处理示例 ===")
    
    # 创建示例Markdown文件
    sample_content = """
# 人工智能简介

人工智能（AI）是计算机科学的一个分支。它致力于创建能够执行通常需要人类智能的任务的系统。

## 机器学习

机器学习是AI的一个子领域。它使计算机能够在没有明确编程的情况下学习和改进。

### 监督学习

监督学习使用标记的训练数据来学习映射函数。常见的算法包括线性回归和决策树。

### 无监督学习

无监督学习从未标记的数据中发现隐藏的模式。聚类和降维是常见的无监督学习任务。

## 深度学习

深度学习使用多层神经网络来模拟人脑的工作方式。它在图像识别和自然语言处理方面取得了突破性进展。
    """
    
    # 保存示例文件
    example_file = "example_ai_intro.md"
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    # 初始化处理器
    processor = SentenceEmbeddingProcessor(
        model_name="all-MiniLM-L6-v2",
        db_path="batch_example.db"
    )
    
    # 批量处理文件
    print(f"正在处理文件: {example_file}")
    sentence_count = processor.process_file(example_file, batch_size=16)
    print(f"处理完成，共 {sentence_count} 个句子")
    
    # 构建索引
    search_index = SimilaritySearchIndex(
        db_path="batch_example.db",
        index_path="batch_example_index.faiss"
    )
    search_index.build_index()
    
    # 测试搜索
    queries = ["什么是机器学习", "神经网络的应用", "无监督学习算法"]
    
    for query in queries:
        print(f"\n查询: '{query}'")
        query_embedding = processor.encode_query(query)
        results = search_index.search(query_embedding, k=2, threshold=0.1)
        
        if results:
            for result in results:
                print(f"  相似度: {result['similarity_score']:.4f}")
                print(f"  句子: {result['text'][:100]}...")
        else:
            print("  未找到相关结果")
    
    # 清理示例文件
    if os.path.exists(example_file):
        os.remove(example_file)

def example_database_operations():
    """数据库操作示例"""
    print("=== 数据库操作示例 ===")
    
    db = EmbeddingDatabase("operations_example.db")
    
    # 添加文档
    doc_id1 = db.add_document("doc1.md", "/path/to/doc1.md")
    doc_id2 = db.add_document("doc2.md", "/path/to/doc2.md")
    
    print(f"添加文档，ID: {doc_id1}, {doc_id2}")
    
    # 查看统计信息
    stats = db.get_stats()
    print(f"数据库统计: {stats}")
    
    # 获取文档句子（如果存在）
    sentences = db.get_document_sentences(doc_id1)
    print(f"文档 {doc_id1} 的句子数量: {len(sentences)}")

def example_custom_search():
    """自定义搜索示例"""
    print("=== 自定义搜索示例 ===")
    
    # 使用现有的数据库和索引
    if not os.path.exists("example_embeddings.db"):
        print("请先运行基本使用示例来创建数据")
        return
    
    search_index = SimilaritySearchIndex(
        db_path="example_embeddings.db",
        index_path="example_index.faiss"
    )
    
    processor = SentenceEmbeddingProcessor(
        model_name="all-MiniLM-L6-v2",
        db_path="example_embeddings.db"
    )
    
    # 多个查询的批量搜索
    queries = [
        "机器学习算法",
        "神经网络架构", 
        "人工智能应用",
        "数据处理方法"
    ]
    
    print("批量搜索结果:")
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. 查询: '{query}'")
        query_embedding = processor.encode_query(query)
        results = search_index.search(query_embedding, k=2, threshold=0.2)
        
        if results:
            for j, result in enumerate(results, 1):
                print(f"   {j}. 相似度: {result['similarity_score']:.4f}")
                print(f"      句子: {result['text']}")
        else:
            print("   未找到相关结果")

def cleanup_example_files():
    """清理示例文件"""
    example_files = [
        "example_embeddings.db",
        "example_index.faiss",
        "example_index_mapping.pkl",
        "batch_example.db",
        "batch_example_index.faiss",
        "batch_example_index_mapping.pkl",
        "operations_example.db",
        "example_ai_intro.md"
    ]
    
    print("清理示例文件...")
    for file in example_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"已删除: {file}")

def main():
    """主函数"""
    print("句子嵌入系统使用示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        example_basic_usage()
        print("\n" + "=" * 50)
        
        example_batch_processing()
        print("\n" + "=" * 50)
        
        example_database_operations()
        print("\n" + "=" * 50)
        
        example_custom_search()
        print("\n" + "=" * 50)
        
        # 询问是否清理文件
        response = input("是否清理示例文件？(y/N): ")
        if response.lower() == 'y':
            cleanup_example_files()
        
        print("\n示例运行完成！")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()