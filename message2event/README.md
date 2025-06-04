# 模块2数据、代码说明

## 1.数据使用说明

模块2输出的csv文件：**data_bert.csv**--微调bert预测的事件类别；**data_rules.csv**--基于关键词/规则判断得到的事件类别

数据处理逻辑：基于模块1输出的清洗后的csv文件，预测message对应的事件类别、情感分数、可靠性；

事件类别：对应csv文件中的"event_type"列，总共有6种事件类别--power issues、water issues、road issues、medical issues、building damage issues、Other complaints

情感分数：对应对应csv文件中的"sentiment_score"列，取值范围 0~1，越低越负面

可靠性：对应对应csv文件中的"reliability"列，总共有Reliable、Unreliable、Neutral三个类别

**关于如何使用：**可靠性判断为Unreliable、Neutral对应行的数据，大都是与灾情无关的广告/闲聊数据，分析的时候可以忽略。所以实际使用的时候，直接提取可靠性为Reliable的数据即可

## 2.脚本功能说明

sentiment.py：预测情感分数的脚本

reliable.py：判断可靠性的脚本

sample_data.py：采样训练bert数据的脚本

train_bert.py：训练bert的脚本

predict_event_bert.py：用bert预测事件类别的脚本

predict_event_rules.py：用关键词/规则预测事件类别的脚本

