import pandas as pd
import re

def classify_event(message):
    """
    基于关键词规则对消息进行事件分类
    
    Args:
        message (str): 待分类的消息文本
        
    Returns:
        str: 事件类型
    """
    # 扩充的事件类型和关键词表
    event_keywords = {
        # 电力问题：涉及电力中断、供电设备、核电站等
        'power issues': [
            # 基础电力词汇
            'power', 'electricity', 'electric', 'electrical', 'energy',
            'nuclear', 'plant', 'grid', 'outage', 'blackout', 'brownout',
            'generator', 'transformer', 'circuit', 'voltage', 'current',
            'station', 'generation', 'nuclear plant', 'power plant',
            
            # 故障相关
            'power failure', 'power cut', 'power down', 'power loss',
            'electrical failure', 'grid failure', 'system failure',
            'no power', 'lost power', 'without power', 'power restored',
            
            # 核电相关
            'radiation', 'reactor', 'atomic', 'radioactive', 'nuclear waste',
            'meltdown', 'leak', 'contamination', 'nuclear accident',
            'nuclear emergency', 'radiation leak', 'nuclear crisis',
            
            # 设备相关
            'power line', 'power cable', 'electrical wire', 'utility pole',
            'substation', 'power grid', 'transmission', 'distribution',
            'electrical system', 'power supply', 'backup power',
            
            # 修复相关
            'electrician', 'repair crew', 'utility company', 'power company',
            'restore power', 'fix power', 'electrical repair'
        ],
        
        # 水务问题：供水中断、水质污染、水管设施等
        'water issues': [
            # 基础水务词汇
            'water', 'pipe', 'pipeline', 'plumbing', 'drain', 'drainage',
            'sewer', 'sewage', 'reservoir', 'tank', 'well', 'pump',
            'supply', 'treatment', 'purification', 'filtration',
            
            # 水质问题
            'contamination', 'contaminated', 'polluted', 'pollution',
            'dirty water', 'bad water', 'unsafe water', 'toxic water',
            'water quality', 'drinking water', 'tap water', 'clean water',
            'water test', 'water sample', 'water analysis',
            
            # 供水问题
            'no water', 'water shortage', 'water cut', 'water outage',
            'low pressure', 'no pressure', 'water pressure', 'water flow',
            'water main', 'main break', 'pipe burst', 'water leak',
            'flooding', 'flood', 'overflow', 'backup',
            
            # 设施相关
            'faucet', 'tap', 'hydrant', 'fire hydrant', 'valve',
            'water meter', 'irrigation', 'sprinkler', 'water system',
            'water treatment plant', 'sewage treatment', 'waste water',
            
            # 干旱相关
            'drought', 'dry', 'shortage', 'rationing', 'conservation',
            
            # 修复相关
            'plumber', 'water department', 'utility', 'repair water',
            'fix pipe', 'replace pipe'
        ],
        
        # 道路/交通问题：交通事故、道路损坏、拥堵等
        'road issues': [
            # 基础交通词汇
            'traffic', 'road', 'highway', 'freeway', 'street', 'avenue',
            'boulevard', 'lane', 'intersection', 'crosswalk', 'sidewalk',
            'bridge', 'tunnel', 'overpass', 'underpass', 'ramp',
            
            # 交通事故
            'accident', 'crash', 'collision', 'wreck', 'hit', 'struck',
            'car accident', 'auto accident', 'vehicle accident',
            'fender bender', 'rear end', 'head on', 'rollover',
            'overturned', 'flipped', 'totaled', 'damaged',
            
            # 道路问题
            'pothole', 'crack', 'damage', 'repair', 'construction',
            'road work', 'maintenance', 'resurfacing', 'paving',
            'closed road', 'road closure', 'detour', 'bypass',
            'blocked', 'barricade', 'cone', 'barrier',
            
            # 交通拥堵
            'congestion', 'jam', 'traffic jam', 'gridlock', 'backup',
            'slow traffic', 'heavy traffic', 'rush hour', 'delay',
            'stuck', 'standstill', 'crawling', 'bumper to bumper',
            
            # 交通信号
            'signal', 'light', 'traffic light', 'stop light', 'red light',
            'green light', 'traffic signal', 'broken light', 'out of order',
            
            # 驾驶行为
            'reckless', 'speeding', 'cut off', 'road rage', 'aggressive',
            'drunk driving', 'impaired', 'distracted', 'texting',
            
            # 车辆相关
            'vehicle', 'car', 'truck', 'bus', 'motorcycle', 'bike',
            'emergency vehicle', 'ambulance', 'fire truck', 'police car',
            
            # 通勤相关
            'commute', 'commuter', 'rush', 'travel', 'route', 'alternate route'
        ],
        
        # 医疗/求助问题：健康危机、紧急救助、医疗服务等
        'medical issues': [
            # 医疗机构
            'hospital', 'clinic', 'emergency room', 'er', 'urgent care',
            'medical center', 'health center', 'doctor office', 'pharmacy',
            
            # 医疗人员
            'doctor', 'physician', 'nurse', 'paramedic', 'emt',
            'surgeon', 'specialist', 'therapist', 'medic',
            
            # 紧急情况
            'emergency', 'urgent', '911', 'call 911', 'help', 'rescue',
            'crisis', 'critical', 'life threatening', 'serious condition',
            'ambulance', 'medical emergency', 'health emergency',
            
            # 伤病症状
            'sick', 'illness', 'disease', 'injury', 'injured', 'hurt',
            'pain', 'ache', 'fever', 'cough', 'bleeding', 'bleed',
            'wound', 'cut', 'burn', 'broken', 'fracture', 'sprain',
            'unconscious', 'faint', 'dizzy', 'nausea', 'vomit',
            
            # 严重疾病
            'heart attack', 'stroke', 'seizure', 'overdose', 'poisoning',
            'allergic reaction', 'asthma', 'diabetic', 'cardiac',
            'respiratory', 'breathing', 'chest pain',
            
            # 求助相关
            'help needed', 'need help', 'assistance', 'aid', 'support',
            'relief', 'rescue', 'save', 'emergency contact',
            
            # 失踪/危险
            'missing', 'lost', 'danger', 'dangerous', 'hazard', 'hazardous',
            'safety', 'unsafe', 'risk', 'threat', 'warning',
            
            # 灾难相关
            'disaster', 'catastrophe', 'tragedy', 'evacuation', 'evacuate',
            'shelter', 'refugee', 'victim', 'casualty', 'survivor',
            
            # 健康相关
            'health', 'medical', 'treatment', 'medication', 'surgery',
            'operation', 'procedure', 'diagnosis', 'condition', 'symptom'
        ],
        
        # 建筑受损问题：建筑损坏、结构问题、地震影响等
        'building damage issues': [
            # 建筑结构
            'building', 'house', 'home', 'apartment', 'condo', 'office',
            'structure', 'construction', 'foundation', 'wall', 'ceiling',
            'floor', 'roof', 'window', 'door', 'stairs', 'balcony',
            
            # 损坏相关
            'damage', 'damaged', 'broken', 'crack', 'cracked', 'split',
            'collapse', 'collapsed', 'fall', 'fell', 'fallen', 'lean',
            'unsafe', 'unstable', 'structural damage', 'foundation damage',
            
            # 地震相关
            'earthquake', 'quake', 'tremor', 'aftershock', 'seismic',
            'shake', 'shaking', 'shook', 'vibrate', 'vibration',
            'tremble', 'trembling', 'epicenter', 'magnitude', 'richter',
            
            # 破坏程度
            'destroyed', 'demolition', 'demolished', 'ruins', 'rubble',
            'debris', 'wreckage', 'shattered', 'glass', 'broken glass',
            'structural failure', 'building failure', 'collapse',
            
            # 维修相关
            'repair', 'fix', 'restoration', 'rebuild', 'reconstruct',
            'renovate', 'contractor', 'construction crew', 'engineer',
            
            # 安全相关
            'evacuation', 'evacuate', 'condemned', 'uninhabitable',
            'safety hazard', 'structural integrity', 'inspection',
            'building code', 'violation', 'permit',
            
            # 财产相关
            'property', 'real estate', 'insurance', 'claim', 'assessment',
            'loss', 'damage assessment', 'structural engineer'
        ]
    }
    
    # 如果消息为空或不是字符串，返回其他无用信息
    if not isinstance(message, str) or not message.strip():
        return 'Other complaints'
    
    # 将消息转为小写以便匹配
    message_lower = message.lower()
    
    # 记录每个类别的匹配得分
    category_scores = {}
    
    # 遍历各个事件类型
    for category, keywords in event_keywords.items():
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            # 使用正则表达式进行全词匹配，避免部分匹配
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = re.findall(pattern, message_lower)
            if matches:
                score += len(matches)
                matched_keywords.extend(matches)
        
        category_scores[category] = {
            'score': score,
            'keywords': matched_keywords
        }
    
    # 找到得分最高的类别
    max_score = max([scores['score'] for scores in category_scores.values()])
    
    # 如果没有任何关键词匹配，返回其他无用信息
    if max_score == 0:
        return 'Other complaints'
    
    # 返回得分最高的类别
    for category, scores in category_scores.items():
        if scores['score'] == max_score:
            return category
    
    return 'Other complaints'

def process_csv(input_file, output_file):
    """
    处理CSV文件，对消息进行事件分类
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查是否存在message列
        if 'message' not in df.columns:
            print("错误：CSV文件中未找到'message'列")
            return
        
        print(f"开始处理{len(df)}条记录...")
        
        # 对每条消息进行事件分类
        df['event_type'] = df['message'].apply(classify_event)
        
        # 统计各类别数量
        category_counts = df['event_type'].value_counts()
        print("\n分类结果统计：")
        for category, count in category_counts.items():
            print(f"{category}: {count}条")
        
        # 计算分类分布百分比
        print("\n分类分布百分比：")
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{category}: {percentage:.2f}%")
        
        # 保存结果到新的CSV文件
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n分类完成！结果已保存到: {output_file}")
        
        # 显示一些具体分类的示例
        print("\n各类别分类示例：")
        for category in df['event_type'].unique():
            if category != 'Other complaints':
                examples = df[df['event_type'] == category]['message'].head(3)
                print(f"\n{category}类别示例:")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")

def main():
    """主函数"""
    # 设置输入和输出文件路径
    input_file = "/Users/wangrunze/Desktop/可视化pj-社交媒体分析/output_with_sentiment.csv"
    output_file = "output_with_event_type_enhanced.csv"
    
    # 处理CSV文件
    process_csv(input_file, output_file)

if __name__ == "__main__":
    main()