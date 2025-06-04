import re

def classify_reliability(message, sentiment_score):
    """
    基于规则对消息进行可靠性分类。
    利用城市背景信息、消息特征和情感得分。
    """

    # 处理空值情况
    if pd.isna(message) or message is None or str(message).strip() == '':
        return 'Neutral'
    
    # 处理情感得分空值
    if pd.isna(sentiment_score):
        sentiment_score = 0.5  # 默认中性情感得分
        
    message_lower = message.lower()

    # 1. 优先识别明确的可靠信息特征
    # 地震相关关键词（结合上下文）
    earthquake_keywords = ['earthquake', 'aftershock', 'tremor']
    earthquake_context = ['felt', 'hit', 'struck', 'magnitude', 'epicenter', 'damage from']
    
    reliable_keywords_tmp = [# 基础电力词汇
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
            'restore power', 'fix power', 'electrical repair',

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
            'fix pipe', 'replace pipe',

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
            'commute', 'commuter', 'rush', 'travel', 'route', 'alternate route',

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
            'operation', 'procedure', 'diagnosis', 'condition', 'symptom',

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
            'loss', 'damage assessment', 'structural engineer']

    reliable_keywords_tmp = reliable_keywords_tmp + earthquake_keywords 
    reliable_keywords = list(set(reliable_keywords_tmp))  # 去重


    
    # 检查是否为可靠信息
    if (any(keyword in message_lower for keyword in reliable_keywords)):
        # 进一步验证：检查是否同时包含垃圾信息特征
        if not _contains_spam_features(message_lower):
            return 'Reliable'
    
    # 特殊情况：地震感受词汇需要上下文验证
    earthquake_feeling_words = ['shaking', 'trembling', 'quaking', 'vibrating']
    if any(word in message_lower for word in earthquake_feeling_words):
        # 如果与地震上下文相关，且没有营销词汇，则可能可靠
        if (any(context in message_lower for context in earthquake_context + ['felt', 'ground', 'building']) and
            not any(spam in message_lower for spam in ['opportunity', 'deal', 'sale', 'advantage'])):
            return 'Reliable'

    # 2. 识别明确的垃圾信息
    if _is_spam_message(message_lower, sentiment_score):
        return 'Unreliable'

    # 3. 默认为中性
    return 'Neutral'


def _contains_spam_features(message_lower):
    """检查是否包含垃圾信息特征"""
    # 明确的垃圾关键词
    definite_spam_keywords_tmp = [
        # 营销推广类
        'opportunity', 'opportunities', 'sales', 'deal', 'deals', 'discount', 
        'free money', 'win now', 'click here', 'take advantage', 'grip',
        'grab', 'clasp', 'hold onto', 'exciting sale', 'thrilling sale', 
        'dramatic sale', 'breathtaking deals', 'impressive deals', 'appealing',
        'vibrating', 'moving', 'trembling',
        
        # 拼写错误/垃圾文本特征
        'bedoreefore', 'advantheeseage', 'ytouou', 'theeserembling', 'theeseh',
        'vibedorerating', 'billeeeeer', 'misconduct', 'opportahtunitahties',
        'tahthataht', 'vibratahte', 
        
        # 侮辱性词汇
        'apple-john', 'barnacle', 'maggot-pies', 'sheep-biting', 'foot-lickers',
        'fustilarian', 'varseals', 'bugbear', 'moldwarp', 'flax-wenchs',
        'gudgeons', 'harpy', 'scut', 'ratsbane', 'lewdster', 'giglet',
        'flap-dragon', 'malt-worm', 'lout',
        
        # 明显的垃圾信息模式
        'change your life', 'make your day', 'make your week', 'make your weekend',
        'by the horns', 'by the throat', 'by the wheel', 'by the collar',
        'short hairs', 'forelock',
    ]

    definite_spam_keywords = list(set(definite_spam_keywords_tmp))  # 去重
    
    if any(keyword in message_lower for keyword in definite_spam_keywords):
        return True
    
    # 营销组合词汇
    marketing_combinations = [
        ('take advantage', 'opportunity'),
        ('grab', 'opportunity'),
        ('days left', 'opportunity'),
        ('exciting', 'deal'),
        ('thrilling', 'sale')
    ]
    
    for combo in marketing_combinations:
        if all(word in message_lower for word in combo):
            return True
    
    return False


def _is_spam_message(message_lower, sentiment_score):
    """判断是否为垃圾信息"""
    
    # 1. 明确的垃圾关键词组合
    definite_spam_keywords_tmp = [
        # 营销推广类
        'opportunity', 'opportunities', 'sales', 'deal', 'deals', 'discount', 
        'free money', 'win now', 'click here', 'take advantage', 'grip',
        'grab', 'clasp', 'hold onto', 'exciting sale', 'thrilling sale', 
        'dramatic sale', 'breathtaking deals', 'impressive deals', 'appealing',
        'vibrating', 'moving', 'trembling',
        
        # 拼写错误/垃圾文本特征
        'bedoreefore', 'advantheeseage', 'ytouou', 'theeserembling', 'theeseh',
        'vibedorerating', 'billeeeeer', 'misconduct', 'opportahtunitahties',
        'tahthataht', 'vibratahte', 
        
        # 侮辱性词汇
        'apple-john', 'barnacle', 'maggot-pies', 'sheep-biting', 'foot-lickers',
        'fustilarian', 'varseals', 'bugbear', 'moldwarp', 'flax-wenchs',
        'gudgeons', 'harpy', 'scut', 'ratsbane', 'lewdster', 'giglet',
        'flap-dragon', 'malt-worm', 'lout',
        
        # 明显的垃圾信息模式
        'change your life', 'make your day', 'make your week', 'make your weekend',
        'by the horns', 'by the throat', 'by the wheel', 'by the collar',
        'short hairs', 'forelock',
    ]
    definite_spam_keywords = list(set(definite_spam_keywords_tmp))  # 去重
    
    # 检查是否匹配多个垃圾关键词
    spam_score = sum(1 for keyword in definite_spam_keywords if keyword in message_lower)
    if spam_score >= 2:
        return True
    
    # 2. 检查垃圾信息句式模式
    spam_patterns = [
        r'take advantage of.*opportunity',
        r'grab.*by the (horns|throat|wheel)',
        r'only \d+ (days?|weeks?|months?) left.*before.*loose',
        r'start using.*rumble.*right (now|away)',
        r'definitely (need to|worth).*rumble',
        r'make your (day|week|weekend|life).*exciting'
    ]
    
    if any(re.search(pattern, message_lower) for pattern in spam_patterns):
        return True
    
    # 3. 检查重复字符或过多标点
    if (re.search(r'(.)\1{3,}', message_lower) or 
        message_lower.count('!') > 3 or
        message_lower.count('?') > 3):
        return True
    
    # 4. 高情感得分但明显是营销内容
    marketing_words = ['sale', 'deal', 'opportunity', 'discount', 'free', 'win']
    if (sentiment_score > 0.85 and 
        sum(1 for word in marketing_words if word in message_lower) >= 2):
        return True
    
    # 5. rumble app垃圾推广（非官方使用）
    rumble_spam_patterns = [
        r'(download|get|use) rumble',
        r'rumble app is (great|fantastic|amazing)',
        r'definitely.*download.*rumble',
        r'need to.*rumble.*after.*quake'
    ]
    
    if any(re.search(pattern, message_lower) for pattern in rumble_spam_patterns):
        return True
    
    return False


if __name__ == "__main__":
    input_csv = "/Users/wangrunze/Desktop/可视化pj-社交媒体分析/module2/output_with_event_type_enhanced.csv"
    output_csv = "/Users/wangrunze/Desktop/可视化pj-社交媒体分析/module2/output_with_event_type_reliability.csv"

    # 读取CSV文件
    import pandas as pd
    try:
        df = pd.read_csv(input_csv)
        
        # 检查是否存在message列
        if 'message' not in df.columns or 'sentiment_score' not in df.columns:
            print("错误：CSV文件中未找到'message'或'sentiment_score'列")
        else:
            # 对每条消息进行可靠性分类
            df['reliability'] = df.apply(lambda row: classify_reliability(row['message'], row['sentiment_score']), axis=1)
            
            # 保存结果到新的CSV文件
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"可靠性分类完成！结果已保存到: {output_csv}")
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_csv}")
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")