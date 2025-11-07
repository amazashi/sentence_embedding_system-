#!/usr/bin/env python3
"""
句子嵌入批处理和索引构建系统
使用 all-MiniLM-L6-v2 模型进行句子嵌入，支持批量处理和相似性搜索
"""

import os
import sys
import logging
import argparse
from typing import List, Dict
import json

from embedding_processor import SentenceEmbeddingProcessor
from search_index import SimilaritySearchIndex
from database import EmbeddingDatabase

def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('embedding_system.log', encoding='utf-8')
        ]
    )

def process_files(args):
    """批量处理文件"""
    processor = SentenceEmbeddingProcessor(
        model_name=args.model,
        db_path=args.database
    )
    
    if args.file:
        # 处理单个文件
        count = processor.process_file(args.file, args.batch_size)
        print(f"处理完成：{count} 个句子")
    elif args.directory:
        # 处理目录
        results = processor.process_directory(
            args.directory, 
            args.pattern, 
            args.batch_size
        )
        
        total = sum(results.values())
        print(f"批量处理完成：")
        print(f"  - 处理文件数：{len(results)}")
        print(f"  - 总句子数：{total}")
        
        if args.verbose:
            for file_path, count in results.items():
                print(f"  - {os.path.basename(file_path)}: {count} 句子")
    else:
        print("错误：请指定要处理的文件或目录")
        return

def build_index(args):
    """构建搜索索引"""
    search_index = SimilaritySearchIndex(
        db_path=args.database,
        index_path=args.index_file
    )
    
    success = search_index.build_index(args.index_type)
    if success:
        stats = search_index.get_index_stats()
        print("索引构建成功：")
        print(f"  - 向量数量：{stats['total_vectors']}")
        print(f"  - 向量维度：{stats['dimension']}")
        print(f"  - 索引类型：{stats['index_type']}")
    else:
        print("索引构建失败")

def search_similar(args):
    """搜索相似句子"""
    # 初始化处理器和搜索索引
    processor = SentenceEmbeddingProcessor(
        model_name=args.model,
        db_path=args.database
    )
    
    search_index = SimilaritySearchIndex(
        db_path=args.database,
        index_path=args.index_file
    )
    
    # 编码查询
    query_embedding = processor.encode_query(args.query)
    
    # 搜索
    results = search_index.search(
        query_embedding, 
        k=args.top_k, 
        threshold=args.threshold
    )
    
    if not results:
        print("未找到相似的句子")
        return
    
    print(f"找到 {len(results)} 个相似句子：\n")
    
    for result in results:
        print(f"排名 {result['rank']} (相似度: {result['similarity_score']:.4f})")
        print(f"文件: {result['filename']}")
        print(f"句子: {result['text'][:200]}...")
        print("-" * 80)

def show_stats(args):
    """显示数据库统计信息"""
    db = EmbeddingDatabase(args.database)
    stats = db.get_stats()
    
    print("数据库统计信息：")
    print(f"  - 文档数量：{stats['documents']}")
    print(f"  - 句子数量：{stats['sentences']}")
    print(f"  - 嵌入向量数量：{stats['embeddings']}")
    
    # 索引统计
    search_index = SimilaritySearchIndex(
        db_path=args.database,
        index_path=args.index_file
    )
    
    index_stats = search_index.get_index_stats()
    if index_stats:
        print(f"  - 索引向量数量：{index_stats['total_vectors']}")
        print(f"  - 索引类型：{index_stats['index_type']}")

def clear_data(args):
    """清空数据库和索引"""
    if not args.confirm:
        response = input("确定要清空所有数据吗？(y/N): ")
        if response.lower() != 'y':
            print("操作已取消")
            return
    
    # 清空数据库
    db = EmbeddingDatabase(args.database)
    db.clear_database()
    
    # 删除索引文件
    index_files = [args.index_file, args.index_file.replace('.faiss', '_mapping.pkl')]
    for file_path in index_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除索引文件：{file_path}")
    
    print("数据清空完成")

def main():
    parser = argparse.ArgumentParser(
        description="句子嵌入批处理和索引构建系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 处理单个文件
  python main.py process --file document.md
  
  # 批量处理目录
  python main.py process --directory ./docs --pattern "*.md"
  
  # 构建搜索索引
  python main.py build-index
  
  # 搜索相似句子
  python main.py search --query "机器学习算法"
  
  # 查看统计信息
  python main.py stats
        """
    )
    
    # 全局参数
    parser.add_argument('--model', default='all-MiniLM-L6-v2', 
                       help='句子嵌入模型名称')
    parser.add_argument('--database', default='embeddings.db', 
                       help='SQLite数据库文件路径')
    parser.add_argument('--index-file', default='similarity_index.faiss', 
                       help='FAISS索引文件路径')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--verbose', action='store_true', 
                       help='显示详细输出')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 处理文件命令
    process_parser = subparsers.add_parser('process', help='处理文件生成嵌入')
    process_group = process_parser.add_mutually_exclusive_group(required=True)
    process_group.add_argument('--file', help='要处理的单个文件')
    process_group.add_argument('--directory', help='要处理的目录')
    process_parser.add_argument('--pattern', default='*.md', 
                               help='文件匹配模式 (默认: *.md)')
    process_parser.add_argument('--batch-size', type=int, default=32, 
                               help='批处理大小 (默认: 32)')
    
    # 构建索引命令
    index_parser = subparsers.add_parser('build-index', help='构建搜索索引')
    index_parser.add_argument('--index-type', default='flat', 
                             choices=['flat', 'ivf'],
                             help='索引类型 (默认: flat)')
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='搜索相似句子')
    search_parser.add_argument('--query', required=True, help='搜索查询')
    search_parser.add_argument('--top-k', type=int, default=10, 
                              help='返回结果数量 (默认: 10)')
    search_parser.add_argument('--threshold', type=float, default=0.0, 
                              help='相似度阈值 (默认: 0.0)')
    
    # 统计信息命令
    subparsers.add_parser('stats', help='显示数据库统计信息')
    
    # 清空数据命令
    clear_parser = subparsers.add_parser('clear', help='清空所有数据')
    clear_parser.add_argument('--confirm', action='store_true', 
                             help='跳过确认提示')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 设置日志
    setup_logging(args.log_level)
    
    try:
        if args.command == 'process':
            process_files(args)
        elif args.command == 'build-index':
            build_index(args)
        elif args.command == 'search':
            search_similar(args)
        elif args.command == 'stats':
            show_stats(args)
        elif args.command == 'clear':
            clear_data(args)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
    except Exception as e:
        logging.error(f"执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()