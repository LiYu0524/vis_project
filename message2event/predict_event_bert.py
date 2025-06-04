import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class PredictionDataset(Dataset):
    """
    用于预测的数据集类
    """
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 使用BERT tokenizer进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }

def predict_csv_events(csv_file_path, model_path, output_csv_path, batch_size=32):
    """
    使用训练好的模型对CSV文件中的消息进行事件分类预测
    
    Args:
        csv_file_path: 输入CSV文件路径
        model_path: 训练好的模型权重文件路径
        output_csv_path: 输出CSV文件路径
        batch_size: 批处理大小
    """
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 读取CSV文件
    print("Loading CSV file...")
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} records")
    
    # 提取消息文本
    messages = df['message'].fillna('').tolist()  # 处理可能的NaN值
    
    # 加载模型权重和配置
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint中获取标签映射
    if 'label_map' in checkpoint:
        label_map = checkpoint['label_map']
        id_to_label = {v: k for k, v in label_map.items()}
    else:
        # 如果checkpoint中没有label_map，使用默认映射
        label_map = {
            'power issues': 0,
            'water issues': 1, 
            'road issues': 2,
            'medical issues': 3,
            'building damage issues': 4,
            'Other complaints': 5
        }
        id_to_label = {v: k for k, v in label_map.items()}
    
    print(f"Label mapping: {label_map}")
    
    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_map),
        output_attentions=False,
        output_hidden_states=False
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # 创建预测数据集和数据加载器
    prediction_dataset = PredictionDataset(messages, tokenizer)
    prediction_loader = DataLoader(
        prediction_dataset,
        batch_size=batch_size,
        shuffle=False,  # 保持顺序
        num_workers=4 if device.type == 'cuda' else 2
    )
    
    # 进行预测
    print("Starting prediction...")
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for batch in tqdm(prediction_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算概率和预测
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            batch_confidences = probabilities.max(dim=1)[0].cpu().numpy()
            
            # 转换为标签名称
            batch_labels = [id_to_label[pred] for pred in batch_predictions]
            
            predictions.extend(batch_labels)
            confidences.extend(batch_confidences)
    
    print(f"Prediction completed! Processed {len(predictions)} messages")
    
    # 更新DataFrame
    df['event_type'] = predictions
    df['prediction_confidence'] = confidences  # 添加置信度列
    
    # 显示预测统计
    print("\nPrediction statistics:")
    print(df['event_type'].value_counts())
    print(f"\nAverage confidence: {np.mean(confidences):.4f}")
    
    # 保存结果
    df.to_csv(output_csv_path, index=False)
    print(f"\nResults saved to: {output_csv_path}")
    
    # 显示一些示例预测
    print("\nSample predictions:")
    sample_df = df.sample(n=min(10, len(df)))
    for idx, row in sample_df.iterrows():
        print(f"Message: {row['message'][:100]}...")
        print(f"Predicted: {row['event_type']} (confidence: {row['prediction_confidence']:.4f})")
        print("-" * 50)
    
    return df

def batch_predict_with_analysis(csv_file_path, model_path, output_csv_path, batch_size=32):
    """
    带分析功能的批量预测函数
    """
    # 执行预测
    df = predict_csv_events(csv_file_path, model_path, output_csv_path, batch_size)
    
    # 分析结果
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    # 置信度分析
    confidence_stats = df['prediction_confidence'].describe()
    print(f"\nConfidence Statistics:")
    print(confidence_stats)
    
    # 低置信度预测
    low_confidence_threshold = 0.5
    low_confidence_predictions = df[df['prediction_confidence'] < low_confidence_threshold]
    print(f"\nLow confidence predictions (< {low_confidence_threshold}): {len(low_confidence_predictions)}")
    
    if len(low_confidence_predictions) > 0:
        print("Sample low confidence predictions:")
        for idx, row in low_confidence_predictions.head(5).iterrows():
            print(f"Message: {row['message'][:80]}...")
            print(f"Predicted: {row['event_type']} (confidence: {row['prediction_confidence']:.4f})")
            print("-" * 40)
    
    # 各类别的平均置信度
    print(f"\nAverage confidence by category:")
    category_confidence = df.groupby('event_type')['prediction_confidence'].agg(['mean', 'count'])
    print(category_confidence.round(4))
    
    return df

if __name__ == "__main__":
    # 设置文件路径
    input_csv = 'output_with_event_type_reliability.csv'
    model_weights = 'best_event_classifier.pt'
    output_csv = 'predicted_events_output.csv'
    
    # 执行预测
    try:
        result_df = batch_predict_with_analysis(
            csv_file_path=input_csv,
            model_path=model_weights,
            output_csv_path=output_csv,
            batch_size=64  # 可以根据GPU内存调整
        )
        
        print(f"\n✅ Prediction completed successfully!")
        print(f"📁 Results saved to: {output_csv}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()