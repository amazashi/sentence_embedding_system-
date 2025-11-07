import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from database import EmbeddingDatabase
from embedding_processor import SentenceEmbeddingProcessor
from search_index import SimilaritySearchIndex
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 在生产环境中应该使用更安全的密钥

# 配置文件上传
UPLOAD_FOLDER = 'init_files'
ALLOWED_EXTENSIONS = {'md', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 配置自动初始化
AUTO_SCAN_DIRECTORY = os.environ.get('AUTO_SCAN_DIRECTORY', './init_files/demo')  # 默认扫描 demo 文件夹
AUTO_BUILD_INDEX = os.environ.get('AUTO_BUILD_INDEX', 'true').lower() == 'true'  # 是否自动构建索引

# 确保上传文件夹和初始化目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUTO_SCAN_DIRECTORY, exist_ok=True)

# 初始化组件
db_manager = EmbeddingDatabase()
embedding_processor = SentenceEmbeddingProcessor()
search_index = SimilaritySearchIndex()

def auto_initialize_system():
    """系统自动初始化：扫描目录并构建索引"""
    try:
        # 检查是否需要自动扫描
        if not AUTO_SCAN_DIRECTORY or not os.path.exists(AUTO_SCAN_DIRECTORY):
            logger.info(f"自动扫描目录不存在或未配置: {AUTO_SCAN_DIRECTORY}")
            return
        
        # 检查数据库是否已有数据
        stats = db_manager.get_stats()
        if stats['sentences'] > 0:
            logger.info(f"数据库已有 {stats['sentences']} 个句子，跳过自动初始化")
            return
        
        logger.info(f"开始自动扫描目录: {AUTO_SCAN_DIRECTORY}")
        
        # 扫描并处理文件
        results = embedding_processor.process_directory(AUTO_SCAN_DIRECTORY, "*.md")
        
        if results:
            total_files = len([f for f in results.values() if f > 0])
            total_sentences = sum(results.values())
            logger.info(f"自动初始化完成：处理了 {total_files} 个文件，共 {total_sentences} 个句子")
            
            # 自动构建索引
            if AUTO_BUILD_INDEX and total_sentences > 0:
                logger.info("开始自动构建搜索索引...")
                try:
                    search_index.build_index()
                    logger.info("搜索索引构建完成")
                except Exception as e:
                    logger.error(f"构建搜索索引失败: {e}")
        else:
            logger.info("未找到可处理的文件")
            
    except Exception as e:
        logger.error(f"自动初始化失败: {e}")

# 执行自动初始化
auto_initialize_system()

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    # 检查是否有文件被上传
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': '没有选择文件'})
    
    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'success': False, 'message': '没有选择文件'})
    
    # 检查是否有空文件名
    if all(file.filename == '' for file in files):
        return jsonify({'success': False, 'message': '没有选择文件'})
    
    processed_files = []
    total_sentences = 0
    
    for file in files:
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # 处理文件
                sentences_processed = embedding_processor.process_file(filepath)
                processed_files.append({
                    'filename': filename,
                    'sentences_processed': sentences_processed
                })
                total_sentences += sentences_processed
            except Exception as e:
                logger.error(f"处理文件 {filename} 时出错: {str(e)}")
                return jsonify({'success': False, 'message': f'处理文件 {filename} 时出错: {str(e)}'})
        else:
            return jsonify({'success': False, 'message': f'不支持的文件格式: {file.filename}'})
    
    if processed_files:
        message = f'成功处理 {len(processed_files)} 个文件，共 {total_sentences} 个句子'
        return jsonify({
            'success': True, 
            'message': message,
            'data': {
                'files_processed': len(processed_files),
                'total_sentences': total_sentences,
                'files': processed_files
            }
        })
    else:
        return jsonify({'success': False, 'message': '没有有效的文件被处理'})

@app.route('/build_index', methods=['POST'])
def build_index():
    """构建搜索索引"""
    try:
        # 检查数据库中是否有嵌入向量
        embeddings = db_manager.get_all_embeddings()
        if not embeddings:
            return jsonify({'success': False, 'message': '没有找到嵌入向量，请先处理文件'})
        
        # 构建索引（build_index方法会自己从数据库获取数据）
        success = search_index.build_index()
        
        if success:
            return jsonify({
                'success': True, 
                'message': f'索引构建成功！包含 {len(embeddings)} 个向量'
            })
        else:
            return jsonify({'success': False, 'message': '索引构建失败'})
            
    except Exception as e:
        logger.error(f"构建索引时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'构建索引时出错: {str(e)}'})

@app.route('/search', methods=['POST'])
def search():
    """搜索相似句子"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 5))
        
        if not query:
            return jsonify({'success': False, 'message': '请输入搜索查询'})
        
        # 加载索引
        if not search_index.load_index():
            return jsonify({'success': False, 'message': '索引未找到，请先构建索引'})
        
        # 编码查询
        query_vector = embedding_processor.encode_query(query)
        
        # 搜索
        search_results = search_index.search(query_vector, top_k)
        
        # 处理搜索结果
        results = []
        for result in search_results:
            results.append({
                'rank': result['rank'],
                'sentence': result['text'],
                'similarity': result['similarity_score'],
                'document': result['filename'],
                'sentence_number': result.get('sentence_index', 0)
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'query': query
        })
    except Exception as e:
        logger.error(f"搜索时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'搜索时出错: {str(e)}'})

@app.route('/stats')
def get_stats():
    """获取系统统计信息"""
    try:
        stats = db_manager.get_stats()
        index_stats = search_index.get_index_stats() if search_index.index is not None else None
        
        return jsonify({
            'success': True,
            'database_stats': stats,
            'index_stats': index_stats
        })
    except Exception as e:
        logger.error(f"获取统计信息时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'获取统计信息时出错: {str(e)}'})

@app.route('/get_file_content', methods=['GET'])
def get_file_content():
    """获取指定文件的markdown内容"""
    try:
        filename = request.args.get('filename', '').strip()
        
        if not filename:
            return jsonify({'success': False, 'message': '请提供文件名'})
        
        # 安全检查：确保文件名不包含路径遍历字符
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'success': False, 'message': '无效的文件名'})
        
        # 在多个可能的目录中查找文件
        search_directories = [
            AUTO_SCAN_DIRECTORY,  # 主要扫描目录
            './init_files/literature_standard',  # 标准文献目录
            './init_files/demo',  # 演示目录
            './init_files',  # 根目录
            './uploads'  # 上传目录
        ]
        
        file_path = None
        for directory in search_directories:
            potential_path = os.path.join(directory, filename)
            if os.path.exists(potential_path) and os.path.isfile(potential_path):
                file_path = potential_path
                break
        
        if not file_path:
            return jsonify({'success': False, 'message': f'文件 {filename} 未找到'})
        
        # 读取文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 如果UTF-8解码失败，尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        
        return jsonify({
            'success': True,
            'filename': filename,
            'content': content,
            'file_path': file_path
        })
        
    except Exception as e:
        logger.error(f"获取文件内容时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'获取文件内容时出错: {str(e)}'})

@app.route('/clear', methods=['POST'])
def clear_data():
    """清空数据库和索引"""
    try:
        db_manager.clear_database()
        # 删除索引文件
        if os.path.exists('similarity_index.faiss'):
            os.remove('similarity_index.faiss')
        if os.path.exists('similarity_index_mapping.pkl'):
            os.remove('similarity_index_mapping.pkl')
        
        return jsonify({'success': True, 'message': '数据已清空'})
    except Exception as e:
        logger.error(f"清空数据时出错: {str(e)}")
        return jsonify({'success': False, 'message': f'清空数据时出错: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)