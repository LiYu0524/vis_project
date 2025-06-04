import pandas as pd
import numpy as np
from sklearn.utils import resample

def sample_data_for_bert(input_file, output_file, samples_per_category=1500):
    """
    从每个事件类型中随机抽取指定数量的数据用于BERT训练
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径  
        samples_per_category (int): 每个类别抽取的样本数量
    """
    try:
        # 读取CSV文件
        print(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file)
        
        # 检查必要的列是否存在
        if 'event_type' not in df.columns:
            print("错误：CSV文件中未找到'event_type'列")
            return
        
        if 'message' not in df.columns:
            print("错误：CSV文件中未找到'message'列")
            return
        
        print(f"原始数据总量: {len(df)}条")
        
        # 统计各类别的数量
        print("\n原始数据各类别分布:")
        category_counts = df['event_type'].value_counts()
        for category, count in category_counts.items():
            print(f"{category}: {count}条")
        
        # 定义目标类别（注意类别名称要与实际数据匹配）
        target_categories = [
            'power issues',
            'water issues', 
            'road issues',
            'medical issues',
            'building damage issues',
            'Other complaints'  # 注意这里可能是'Other complaints'而不是'other complaint'
        ]
        
        # 检查实际存在的类别
        actual_categories = df['event_type'].unique()
        print(f"\n实际数据中的类别: {list(actual_categories)}")
        
        # 调整目标类别列表，只包含实际存在的类别
        available_categories = [cat for cat in target_categories if cat in actual_categories]
        if len(available_categories) < len(target_categories):
            print("警告：部分目标类别在数据中不存在")
            print(f"可用类别: {available_categories}")
        
        # 存储抽样后的数据
        sampled_dfs = []
        
        # 对每个类别进行抽样
        for category in available_categories:
            category_data = df[df['event_type'] == category]
            category_count = len(category_data)
            
            print(f"\n处理类别: {category}")
            print(f"该类别原始数据量: {category_count}条")
            
            if category_count == 0:
                print(f"跳过类别 {category}：无数据")
                continue
            elif category_count < samples_per_category:
                print(f"警告：{category} 类别数据不足 {samples_per_category} 条")
                print(f"将使用重复抽样（bootstrap）来达到目标数量")
                # 使用重复抽样来达到目标数量
                sampled_data = resample(category_data, 
                                      n_samples=samples_per_category, 
                                      random_state=42,
                                      replace=True)
            else:
                # 随机抽样指定数量
                sampled_data = category_data.sample(n=samples_per_category, 
                                                  random_state=42)
            
            sampled_dfs.append(sampled_data)
            print(f"已抽取 {len(sampled_data)} 条数据")
        
        # 合并所有抽样的数据
        if sampled_dfs:
            final_df = pd.concat(sampled_dfs, ignore_index=True)
            
            # 打乱数据顺序
            final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"\n最终数据集信息:")
            print(f"总数据量: {len(final_df)}条")
            
            # 统计最终数据的类别分布
            print("\n最终数据各类别分布:")
            final_category_counts = final_df['event_type'].value_counts()
            for category, count in final_category_counts.items():
                print(f"{category}: {count}条")
            
            # 保存到新的CSV文件
            final_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n数据抽样完成！结果已保存到: {output_file}")
            
            # 显示每个类别的一些示例
            print("\n各类别数据示例:")
            for category in final_df['event_type'].unique():
                examples = final_df[final_df['event_type'] == category]['message'].head(2)
                print(f"\n{category}:")
                for i, example in enumerate(examples, 1):
                    # 限制显示长度
                    display_text = example if len(str(example)) <= 100 else str(example)[:100] + "..."
                    print(f"  {i}. {display_text}")
        else:
            print("错误：没有成功抽取到任何数据")
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")

def analyze_data_distribution(file_path):
    """
    分析数据分布情况
    """
    try:
        df = pd.read_csv(file_path)
        print(f"文件: {file_path}")
        print(f"总数据量: {len(df)}条")
        print("\n各类别分布:")
        category_counts = df['event_type'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{category}: {count}条 ({percentage:.2f}%)")
        return df
    except Exception as e:
        print(f"分析文件时出现错误：{str(e)}")
        return None

def main():
    """主函数"""
    # 设置文件路径
    input_file = "/Users/wangrunze/Desktop/可视化pj-社交媒体分析/module2/output_with_event_type_reliability.csv"
    output_file = "/Users/wangrunze/Desktop/可视化pj-社交媒体分析/module2/output_with_event_type_reliability_sample4bert.csv"
    
    # 先分析原始数据分布
    print("=== 原始数据分析 ===")
    analyze_data_distribution(input_file)
    
    print("\n" + "="*50)
    
    # 执行数据抽样
    print("=== 开始数据抽样 ===")
    sample_data_for_bert(input_file, output_file, samples_per_category=1500)
    
    print("\n" + "="*50)
    
    # 分析抽样后的数据
    print("=== 抽样后数据分析 ===")
    analyze_data_distribution(output_file)

if __name__ == "__main__":
    main()