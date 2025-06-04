import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# 设置随机种子保证结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(42)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SocialMediaDataset(Dataset):
    """
    社交媒体事件分类数据集
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用BERT tokenizer进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加[CLS]和[SEP]标记
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EventClassifier:
    """
    事件分类器类，封装了模型训练和评估的完整流程
    """
    def __init__(self, model_name='bert-base-uncased', num_classes=6, max_length=512):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device
        
        # 初始化tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.device)
        
        # 标签映射
        self.label_map = {
            'power issues': 0,
            'water issues': 1, 
            'road issues': 2,
            'medical issues': 3,
            'building damage issues': 4,
            'Other complaints': 5
        }
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
    def prepare_data(self, csv_file, test_size=0.2, val_size=0.1):
        """
        准备训练、验证和测试数据
        """
        # 读取数据
        df = pd.read_csv(csv_file)
        
        # 数据清洗和预处理
        df = df.dropna(subset=['message', 'event_type'])
        
        # 统计各类别分布
        print("事件类型分布:")
        print(df['event_type'].value_counts())
        
        # 提取文本和标签
        texts = df['message'].tolist()
        labels = [self.label_map[label] for label in df['event_type']]
        
        # 分割数据集：先分出测试集，再从剩余数据中分出验证集
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        val_ratio = val_size / (1 - test_size)  # 正确计算临时数据中验证集占比
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"\n数据集划分:")
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本") 
        print(f"测试集: {len(X_test)} 样本")
        
        # 创建Dataset对象
        self.train_dataset = SocialMediaDataset(X_train, y_train, self.tokenizer, self.max_length)
        self.val_dataset = SocialMediaDataset(X_val, y_val, self.tokenizer, self.max_length)
        self.test_dataset = SocialMediaDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def create_data_loaders(self, batch_size=16):
        """
        创建数据加载器
        """
        cpu_count = os.cpu_count()-1
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cpu_count
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cpu_count
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cpu_count
        )
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def train_epoch(self, dataloader, optimizer, scheduler):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # 将数据移到GPU
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            # 记录损失和预测结果
            total_loss += loss.item()
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """
        评估模型性能
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy, predictions, true_labels
    
    def train(self, epochs=5, batch_size=64, learning_rate=2e-5, save_path='best_model.pt'):
        """
        完整的训练流程
        """
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders(batch_size)
        
        # 设置优化器和学习率调度器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 记录训练过程
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_accuracy = 0
        
        print("开始训练...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # 验证
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_accuracy': best_val_accuracy,
                    'label_map': self.label_map
                }, save_path)
                print(f"保存最佳模型，验证准确率: {best_val_accuracy:.4f}")
        
        # 绘制训练曲线
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        print(f"\n训练完成！最佳验证准确率: {best_val_accuracy:.4f}")
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """
        绘制训练曲线
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 准确率曲线
        ax2.plot(train_accuracies, label='Training Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_model(self, model_path='best_model.pt'):
        """
        测试模型并生成详细报告
        """
        # 加载最佳模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print("使用最佳模型进行测试...")
        
        # 在测试集上评估
        test_loss, test_acc, predictions, true_labels = self.evaluate(self.test_loader)
        
        print(f"测试准确率: {test_acc:.4f}")
        print(f"测试损失: {test_loss:.4f}")
        
        # 生成分类报告
        target_names = [self.id_to_label[i] for i in range(self.num_classes)]
        report = classification_report(true_labels, predictions, target_names=target_names)
        print("\n分类报告:")
        print(report)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(true_labels, predictions, target_names)
        
        return test_acc, report
    
    def plot_confusion_matrix(self, true_labels, predictions, target_names):
        """
        绘制混淆矩阵
        """
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_text(self, text, model_path='best_model.pt'):
        """
        对单个文本进行预测
        """
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 编码文本
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
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
            confidence = probabilities.max().cpu().numpy()
        
        predicted_label = self.id_to_label[prediction]
        
        return predicted_label, confidence

def main():
    """
    主函数：执行完整的训练和测试流程
    """
    # 初始化分类器
    classifier = EventClassifier(
        model_name='bert-base-uncased',  # 可以改为 'bert-base-cased' 或其他BERT变体
        num_classes=6,
        max_length=512
    )
    
    # 准备数据
    csv_file = 'output_with_event_type_reliability_sample4bert.csv'
    train_dataset, val_dataset, test_dataset = classifier.prepare_data(csv_file)
    
    # 训练模型
    train_losses, val_losses, train_accs, val_accs = classifier.train(
        epochs=5,  # 可以根据需要调整
        batch_size=64,  # 根据GPU内存调整
        learning_rate=2e-5,
        save_path='best_event_classifier.pt'
    )
    
    # 测试模型
    test_accuracy, classification_report = classifier.test_model('best_event_classifier.pt')
    
    # 示例预测
    print("\n示例预测:")
    sample_texts = [
        "The power plant is not working properly",
        "Water is flooding the streets",
        "The bridge is closed due to damage",
        "People need medical attention at the hospital",
        "The building collapsed after the earthquake",
        "The service at the restaurant was terrible"
    ]
    
    for text in sample_texts:
        predicted_label, confidence = classifier.predict_text(text, 'best_event_classifier.pt')
        print(f"文本: {text}")
        print(f"预测类别: {predicted_label} (置信度: {confidence:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    main()