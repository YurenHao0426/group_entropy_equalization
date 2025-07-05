#!/usr/bin/env python3
"""
WinoBias数据集探索脚本
用于分析数据集是否符合我们的mask-based gender debiasing需求
"""

from datasets import load_dataset
import pandas as pd
import json
import re
from collections import Counter

def main():
    print("🔍 正在下载WinoBias数据集...")
    
    # WinoBias有4个配置
    configs = ['type1_anti', 'type1_pro', 'type2_anti', 'type2_pro']
    
    all_examples = []
    all_configs_data = {}
    
    for config in configs:
        try:
            print(f"\n📥 正在加载配置: {config}")
            dataset = load_dataset("wino_bias", config)
            all_configs_data[config] = dataset
            
            print(f"✅ {config} 下载成功！")
            
            # 查看数据集结构
            print(f"\n📊 {config} 数据集信息:")
            print(f"- 数据集分片: {list(dataset.keys())}")
            
            for split_name, split_data in dataset.items():
                print(f"- {split_name}: {len(split_data)} 条数据")
                print(f"- 字段: {split_data.column_names}")
                
                # 收集所有文本
                if len(split_data) > 0:
                    examples = split_data[:min(10, len(split_data))]
                    
                    # 查看数据结构并转换tokens为文本
                    print(f"\n📝 {config}-{split_name} 前3个例子:")
                    for i in range(min(3, len(examples))):
                        print(f"\n例子 {i+1}:")
                        
                        # 将tokens转换为文本
                        if 'tokens' in examples:
                            text = ' '.join(examples['tokens'][i])
                            print(f"  文本: {text}")
                            all_examples.append(text)
                        
                        # 显示coreference信息
                        if 'coreference_clusters' in examples:
                            print(f"  指代关系: {examples['coreference_clusters'][i]}")
                    
        except Exception as e:
            print(f"❌ 加载 {config} 失败: {e}")
    
    # 如果有数据，进行分析
    if all_examples:
        analyze_dataset(all_examples, all_configs_data)
    else:
        print("❌ 未找到任何文本数据")

def analyze_dataset(all_examples, configs_data):
    """分析数据集是否适合mask任务"""
    
    print(f"\n🎯 整体适用性分析:")
    print(f"总共 {len(all_examples)} 个例子")
    
    # 查找包含职业的句子
    occupation_keywords = [
        'teacher', 'doctor', 'nurse', 'engineer', 'manager', 'secretary', 
        'lawyer', 'developer', 'designer', 'cook', 'mechanic', 'driver',
        'farmer', 'clerk', 'CEO', 'assistant', 'guard', 'baker', 'analyst',
        'salesperson', 'receptionist', 'auditor', 'carpenter', 'mover',
        'accountant', 'janitor', 'librarian'
    ]
    
    gender_pronouns = ['he', 'she', 'his', 'her', 'him']
    
    suitable_for_masking = []
    mask_examples = []
    
    print(f"\n🔍 分析前20个例子:")
    for i, example in enumerate(all_examples[:20]):
        print(f"{i+1}. {example}")
        
        text_lower = example.lower()
        
        # 检查是否包含职业词汇
        has_occupation = any(occ in text_lower for occ in occupation_keywords)
        
        # 检查是否包含性别代词
        has_gender_pronoun = any(pronoun in text_lower for pronoun in gender_pronouns)
        
        print(f"   职业词汇: {has_occupation}, 性别代词: {has_gender_pronoun}")
        
        # 检查句子结构是否适合转换为mask
        if has_occupation and has_gender_pronoun:
            suitable_for_masking.append(example)
            
            # 创建mask版本
            masked_version = create_mask_template(example)
            if masked_version:
                mask_examples.append({
                    'original': example,
                    'masked': masked_version,
                    'gender_words': extract_gender_words(example)
                })
    
    print(f"\n✅ 适合mask任务的例子数量: {len(suitable_for_masking)}")
    print(f"✅ 成功转换为mask格式的例子: {len(mask_examples)}")
    
    if mask_examples:
        print(f"\n📋 Mask转换示例 (前10个):")
        for i, example in enumerate(mask_examples[:10]):
            print(f"\n{i+1}. 原句: {example['original']}")
            print(f"   Mask: {example['masked']}")
            print(f"   性别词: {example['gender_words']}")
    
    # 分析不同配置的特点
    print(f"\n🔍 各配置数据特点:")
    for config_name, dataset in configs_data.items():
        print(f"\n{config_name}:")
        if dataset and len(dataset) > 0:
            # 处理所有分片的数据
            total_size = 0
            occupation_total = 0
            gender_total = 0
            
            for split_name, split_data in dataset.items():
                total_size += len(split_data)
                
                # 分析前10个例子
                examples = split_data[:min(10, len(split_data))]
                if 'tokens' in split_data.column_names:
                    for tokens in examples['tokens']:
                        text = ' '.join(tokens).lower()
                        if any(occ in text for occ in occupation_keywords):
                            occupation_total += 1
                        if any(pronoun in text for pronoun in gender_pronouns):
                            gender_total += 1
            
            print(f"  - 总数据量: {total_size}")
            print(f"  - 包含职业词汇的例子(前10个/分片): {occupation_total}")
            print(f"  - 包含性别代词的例子(前10个/分片): {gender_total}")
    
    # 保存分析结果
    save_analysis_results(len(all_examples), len(suitable_for_masking), mask_examples, configs_data)

def create_mask_template(text):
    """将句子转换为mask格式"""
    patterns = [
        (r'\bhe\b', '[MASK]'),
        (r'\bshe\b', '[MASK]'),
        (r'\bhis\b', '[MASK]'),
        (r'\bher\b', '[MASK]'),
        (r'\bhim\b', '[MASK]')
    ]
    
    masked_text = text
    replacements_made = 0
    
    for pattern, replacement in patterns:
        if re.search(pattern, masked_text, flags=re.IGNORECASE):
            masked_text = re.sub(pattern, replacement, masked_text, flags=re.IGNORECASE, count=1)
            replacements_made += 1
            break  # 只替换第一个找到的
    
    return masked_text if replacements_made > 0 else None

def extract_gender_words(text):
    """提取文本中的性别词汇"""
    gender_words = []
    patterns = [r'\bhe\b', r'\bshe\b', r'\bhis\b', r'\bher\b', r'\bhim\b']
    
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        gender_words.extend(matches)
    
    return list(set([word.lower() for word in gender_words]))

def save_analysis_results(total_examples, suitable_count, mask_examples, configs_data):
    """保存分析结果"""
    
    # 提取所有文本样例
    all_text_samples = []
    
    for config_name, dataset in configs_data.items():
        for split_name, split_data in dataset.items():
            examples = split_data[:5]  # 每个分片取5个例子
            if 'tokens' in split_data.column_names:
                for tokens in examples['tokens']:
                    text = ' '.join(tokens)
                    all_text_samples.append({
                        'config': config_name,
                        'split': split_name,
                        'text': text
                    })
    
    results = {
        'total_examples': total_examples,
        'suitable_for_masking': suitable_count,
        'mask_conversion_success_rate': len(mask_examples) / max(suitable_count, 1),
        'sample_mask_examples': mask_examples[:15],
        'configs_info': {},
        'all_text_samples': all_text_samples[:20]  # 保存前20个文本样例
    }
    
    # 保存各配置信息
    for config_name, dataset in configs_data.items():
        if dataset:
            config_info = {'splits': {}}
            for split_name, split_data in dataset.items():
                config_info['splits'][split_name] = {
                    'size': len(split_data),
                    'columns': split_data.column_names
                }
            results['configs_info'][config_name] = config_info
    
    with open('winobias_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 详细分析结果已保存到 'winobias_analysis.json'")
    
    # 输出总结
    print(f"\n📊 总结:")
    print(f"- 总例子数: {total_examples}")
    print(f"- 适合mask任务: {suitable_count} ({suitable_count/max(total_examples,1)*100:.1f}%)")
    print(f"- 成功转换率: {len(mask_examples)/max(suitable_count,1)*100:.1f}%")
    
    if len(mask_examples) > 10:
        print(f"✅ WinoBias数据集非常适合我们的需求！")
        print(f"   - 包含大量职业+性别代词的句子")
        print(f"   - 可以轻松转换为[MASK]格式")
        print(f"   - 有pro/anti版本用于对比性别偏见")
    else:
        print(f"⚠️  可用数据较少，可能需要额外处理或寻找其他数据集")

if __name__ == "__main__":
    main() 