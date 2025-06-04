import pandas as pd
import torch
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        初始化情感分析器（使用pipeline方式）
        
        Args:
            model_name: 预训练模型名称，默认使用多语言情感分析模型
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1  # pipeline使用0表示GPU，-1表示CPU
        device_name = "GPU" if self.device == 0 else "CPU"
        logger.info(f"使用设备: {device_name}")
        
        # 使用pipeline方式加载模型
        self.load_model()
    
    def load_model(self):
        """使用pipeline方式加载预训练的BERT模型"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            
            # 使用pipeline加载情感分析模型
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
                return_all_scores=True,  # 返回所有类别的概率
                padding=True,  # 启用填充
                truncation=True  # 启用截断
            )
            
            logger.info("模型加载成功!")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def predict_sentiment(self, text, max_length=512):
        """
        预测单个文本的情感得分（用于测试单个样本）
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            
        Returns:
            float: 情感得分 (0-1, 越低越负面)
        """
        if pd.isna(text) or str(text).strip() == "":
            return 0.5  # 对于空文本返回中性得分
        
        try:
            # 截断过长的文本
            text = str(text)
            if len(text) > max_length:
                text = text[:max_length]
            
            # 使用pipeline进行预测
            results = self.sentiment_pipeline(text)
            
            # 提取所有类别的概率
            predictions = [score['score'] for score in results[0]]
            
            # 将预测结果转换为0-1的情感得分
            sentiment_score = self._convert_to_sentiment_score(predictions)
            return float(sentiment_score)
            
        except Exception as e:
            logger.warning(f"处理文本时出错: {e}")
            return 0.5
    
    def predict_batch(self, texts, max_length=512):
        """
        真正的批量预测函数，利用pipeline的批量处理能力
        
        Args:
            texts: 文本列表
            max_length: 最大序列长度
            
        Returns:
            list: 情感得分列表
        """
        if not texts:
            return []
        
        # 预处理文本：处理空值和空字符串
        processed_texts = []
        empty_indices = set()  # 记录空文本的位置
        
        for i, text in enumerate(texts):
            if pd.isna(text) or str(text).strip() == "":
                processed_texts.append("neutral text")  # 用中性文本替代空文本
                empty_indices.add(i)
            else:
                text_str = str(text)
                # 截断过长的文本
                if len(text_str) > max_length:
                    text_str = text_str[:max_length]
                processed_texts.append(text_str)
        
        try:
            # 使用pipeline进行批量预测
            logger.debug(f"开始批量处理 {len(processed_texts)} 个文本")
            batch_results = self.sentiment_pipeline(processed_texts)
            
            scores = []
            for i, result in enumerate(batch_results):
                if i in empty_indices:
                    scores.append(0.5)  # 空文本返回中性得分
                else:
                    # 提取所有类别的概率
                    predictions = [score['score'] for score in result]
                    score = self._convert_to_sentiment_score(predictions)
                    scores.append(float(score))
            
            return scores
            
        except Exception as e:
            logger.warning(f"批量处理时出错: {e}")
            # 如果批量处理失败，回退到单个处理
            logger.info("回退到单个文本处理模式...")
            return [self.predict_sentiment(text, max_length) for text in texts]
    
    def _convert_to_sentiment_score(self, predictions):
        """
        将模型输出转换为0-1的情感得分
        
        Args:
            predictions: 模型输出的概率分布（已经是概率值）
            
        Returns:
            float: 0-1之间的情感得分
        """
        num_labels = len(predictions)
        
        if num_labels == 2:  # 二分类 (负面/正面)
            return predictions[1]  # 正面情感的概率
        elif num_labels == 3:  # 三分类 (负面/中性/正面)
            return (predictions[2] + 0.5 * predictions[1])  # 正面 + 0.5*中性
        elif num_labels == 5:  # 五分类 (1-5星评级)
            # 将1-5星映射到0-1
            weighted_score = sum(predictions[i] * (i + 1) for i in range(5))
            return (weighted_score - 1) / 4  # 归一化到0-1
        else:
            # 其他情况，假设标签按照负面到正面排序
            weighted_score = sum(predictions[i] * i for i in range(num_labels))
            return weighted_score / (num_labels - 1) if num_labels > 1 else 0.5
    
    def analyze_batch(self, texts, batch_size=32, max_length=512):
        """
        批量处理文本，充分利用pipeline的并行计算能力
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            max_length: 最大序列长度
            
        Returns:
            list: 情感得分列表
        """
        all_scores = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # 使用tqdm显示进度条
        with tqdm(total=total_batches, desc="情感分析进度") as pbar:
            for i in range(0, len(texts), batch_size):
                # 获取当前批次的文本
                batch_texts = texts[i:i + batch_size]
                
                # 使用pipeline进行真正的批量处理
                batch_scores = self.predict_batch(batch_texts, max_length)
                
                # 添加到结果列表
                all_scores.extend(batch_scores)
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'batch': f"{len(all_scores)}/{len(texts)}",
                    'avg_score': f"{np.mean(all_scores):.3f}" if all_scores else "0.000"
                })
        
        return all_scores
    
    def analyze_with_memory_management(self, texts, batch_size=32, max_length=512):
        """
        带内存管理的批量处理，防止GPU内存溢出
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            max_length: 最大序列长度
            
        Returns:
            list: 情感得分列表
        """
        all_scores = []
        
        # 动态调整batch_size以适应内存
        current_batch_size = batch_size
        
        with tqdm(total=len(texts), desc="情感分析进度") as pbar:
            i = 0
            while i < len(texts):
                try:
                    # 获取当前批次的文本
                    batch_texts = texts[i:i + current_batch_size]
                    
                    # 批量处理
                    batch_scores = self.predict_batch(batch_texts, max_length)
                    all_scores.extend(batch_scores)
                    
                    # 更新进度
                    i += len(batch_texts)
                    pbar.update(len(batch_texts))
                    pbar.set_postfix({
                        'batch_size': current_batch_size,
                        'processed': f"{i}/{len(texts)}",
                        'avg_score': f"{np.mean(all_scores):.3f}"
                    })
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # GPU内存不足，减小batch_size
                        current_batch_size = max(1, current_batch_size // 2)
                        logger.warning(f"GPU内存不足，调整batch_size为: {current_batch_size}")
                        
                        # 清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise e
        
        return all_scores


def process_csv_file(csv_file_path, text_column='cleanTweet', output_file_path=None, 
                    model_name="nlptown/bert-base-multilingual-uncased-sentiment", 
                    batch_size=32, max_length=512, use_memory_management=True):
    """
    处理CSV文件进行情感分析
    
    Args:
        csv_file_path: 输入CSV文件路径
        text_column: 包含文本的列名
        output_file_path: 输出CSV文件路径，如果为None则自动生成
        model_name: 使用的预训练模型名称
        batch_size: 批处理大小
        max_length: 最大序列长度
        use_memory_management: 是否使用内存管理
        
    Returns:
        str: 输出文件路径
    """
    
    # 读取CSV文件
    logger.info(f"正在读取CSV文件: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
        logger.info(f"成功读取 {len(df)} 行数据")
    except Exception as e:
        logger.error(f"读取CSV文件失败: {e}")
        raise
    
    # 检查列是否存在
    if text_column not in df.columns:
        raise ValueError(f"列 '{text_column}' 不存在于CSV文件中。可用列: {list(df.columns)}")
    
    # 初始化情感分析器
    analyzer = SentimentAnalyzer(model_name=model_name)
    
    # 获取文本数据
    texts = df[text_column].fillna("").astype(str).tolist()
    
    # 进行情感分析
    logger.info(f"开始进行批量情感分析...")
    logger.info(f"总文本数量: {len(texts)}")
    logger.info(f"批处理大小: {batch_size}")
    logger.info(f"最大序列长度: {max_length}")
    
    if use_memory_management:
        sentiment_scores = analyzer.analyze_with_memory_management(
            texts, batch_size=batch_size, max_length=max_length
        )
    else:
        sentiment_scores = analyzer.analyze_batch(
            texts, batch_size=batch_size, max_length=max_length
        )
    
    # 添加情感得分列
    df['sentiment_score'] = sentiment_scores
    
    # 生成输出文件路径
    if output_file_path is None:
        base_name = os.path.splitext(csv_file_path)[0]
        output_file_path = f"{base_name}_with_sentiment.csv"
    
    # 保存结果
    logger.info(f"正在保存结果到: {output_file_path}")
    df.to_csv(output_file_path, index=False)
    logger.info("处理完成!")
    
    # 显示统计信息
    logger.info(f"情感得分统计:")
    logger.info(f"平均值: {np.mean(sentiment_scores):.3f}")
    logger.info(f"标准差: {np.std(sentiment_scores):.3f}")
    logger.info(f"最小值: {np.min(sentiment_scores):.3f}")
    logger.info(f"最大值: {np.max(sentiment_scores):.3f}")
    
    # 情感分布统计
    negative = sum(1 for score in sentiment_scores if score < 0.4)
    neutral = sum(1 for score in sentiment_scores if 0.4 <= score <= 0.6)
    positive = sum(1 for score in sentiment_scores if score > 0.6)
    
    logger.info(f"情感分布:")
    logger.info(f"负面 (< 0.4): {negative} ({negative/len(sentiment_scores)*100:.1f}%)")
    logger.info(f"中性 (0.4-0.6): {neutral} ({neutral/len(sentiment_scores)*100:.1f}%)")
    logger.info(f"正面 (> 0.6): {positive} ({positive/len(sentiment_scores)*100:.1f}%)")
    
    return output_file_path

def test_single_text(text, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    测试单个文本的情感分析
    
    Args:
        text: 输入文本
        model_name: 模型名称
        
    Returns:
        float: 情感得分
    """
    analyzer = SentimentAnalyzer(model_name=model_name)
    score = analyzer.predict_sentiment(text)
    logger.info(f"文本: '{text}'")
    logger.info(f"情感得分: {score:.3f}")
    return score

def test_batch_performance(texts, batch_sizes=[1, 8, 16, 32, 64], 
                          model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    测试不同batch_size的性能
    
    Args:
        texts: 测试文本列表
        batch_sizes: 要测试的批处理大小列表
        model_name: 模型名称
    """
    import time
    
    analyzer = SentimentAnalyzer(model_name=model_name)
    
    logger.info("=== 批处理性能测试 ===")
    logger.info(f"测试文本数量: {len(texts)}")
    
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"\n测试 batch_size = {batch_size}")
        
        start_time = time.time()
        try:
            scores = analyzer.analyze_batch(texts, batch_size=batch_size)
            end_time = time.time()
            
            duration = end_time - start_time
            texts_per_second = len(texts) / duration
            
            results[batch_size] = {
                'duration': duration,
                'texts_per_second': texts_per_second,
                'success': True
            }
            
            logger.info(f"耗时: {duration:.2f}秒")
            logger.info(f"处理速度: {texts_per_second:.2f} 文本/秒")
            
        except Exception as e:
            logger.error(f"batch_size {batch_size} 测试失败: {e}")
            results[batch_size] = {
                'duration': None,
                'texts_per_second': None,
                'success': False,
                'error': str(e)
            }
    
    # 显示性能对比
    logger.info("\n=== 性能对比结果 ===")
    for batch_size, result in results.items():
        if result['success']:
            logger.info(f"batch_size {batch_size:2d}: {result['texts_per_second']:6.2f} 文本/秒")
        else:
            logger.info(f"batch_size {batch_size:2d}: 失败 - {result.get('error', '未知错误')}")
    
    return results

def get_available_models():
    """
    获取可用的预训练情感分析模型列表
    
    Returns:
        dict: 模型名称和描述的字典
    """
    models = {
        "nlptown/bert-base-multilingual-uncased-sentiment": "多语言BERT情感分析（支持中英文）",
        "cardiffnlp/twitter-roberta-base-sentiment-latest": "Twitter专用RoBERTa情感分析",
        "j-hartmann/emotion-english-distilroberta-base": "英文情感和情绪分析",
        "microsoft/DialoGPT-medium": "对话式情感分析",
        "unitary/toxic-bert": "有毒内容检测",
    }
    return models

if __name__ == "__main__":
    # 示例用法
    
    # 显示可用模型
    print("=== 可用模型列表 ===")
    available_models = get_available_models()
    for model_name, description in available_models.items():
        print(f"{model_name}: {description}")
    print()
    
    # 选择模型
    selected_model = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    # 1. 测试单个文本
    print("=== 单文本测试 ===")
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "It's okay, nothing special.",
        "今天天气真好！",
        "这个产品太糟糕了。"
    ]
    
    for text in test_texts:
        test_single_text(text, selected_model)
        print()
    
    # 2. 批处理性能测试
    print("=== 批处理性能测试 ===")
    test_batch_performance(test_texts * 20, model_name=selected_model)
    
    # 3. 处理CSV文件示例
    print("=== CSV文件处理示例 ===")
    
    # 创建示例CSV文件
    sample_data = {
        'id': list(range(1, 101)),  # 创建100个样本
        'cleanTweet': [
            "I love this new feature!",
            "This update is terrible",
            "It's okay, could be better",
            "Amazing work by the team!",
            "Not impressed at all",
            "Great job everyone!",
            "This is disappointing",
            "Fantastic update!",
            "Could use some improvements",
            "Absolutely wonderful!"
        ] * 10,  # 重复10次得到100个样本
        'other_column': [f'Data_{i}' for i in range(1, 101)]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = 'sample_tweets.csv'
    sample_df.to_csv(sample_csv_path, index=False)
    
    # 处理CSV文件
    output_path = process_csv_file(
        csv_file_path=sample_csv_path,
        text_column='cleanTweet',
        model_name=selected_model,
        batch_size=16,  # 使用较大的batch_size来展示并行处理效果
        max_length=128,  # 较短的序列长度以提高速度
        use_memory_management=True
    )
    
    # 显示结果
    result_df = pd.read_csv(output_path)
    print("\n处理结果预览:")
    print(result_df.head(10))
    print(f"\n输出文件已保存到: {output_path}")
    
    # 清理临时文件
    if os.path.exists(sample_csv_path):
        os.remove(sample_csv_path)