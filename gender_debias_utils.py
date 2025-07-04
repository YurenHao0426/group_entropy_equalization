#!/usr/bin/env python3
"""
Gender Debias Utilities
基于GenderBench数据实现性别去偏见训练的工具函数
核心思想：生成除性别外完全一致的回答对
"""

import sys
import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# 添加genderbench到路径
sys.path.append('./genderbench')

class GenderWordProcessor:
    """性别词汇处理器"""
    
    def __init__(self):
        # 性别词汇映射表
        self.gender_pairs = {
            # 基础代词
            'he': 'she', 'she': 'he',
            'him': 'her', 'her': 'him', 
            'his': 'her', 'hers': 'his',
            'himself': 'herself', 'herself': 'himself',
            
            # 群体名词
            'men': 'women', 'women': 'men',
            'man': 'woman', 'woman': 'man',
            'male': 'female', 'female': 'male',
            'males': 'females', 'females': 'males',
            'gentleman': 'lady', 'lady': 'gentleman',
            'gentlemen': 'ladies', 'ladies': 'gentlemen',
            
            # 家庭关系
            'father': 'mother', 'mother': 'father',
            'dad': 'mom', 'mom': 'dad',
            'son': 'daughter', 'daughter': 'son',
            'brother': 'sister', 'sister': 'brother',
            'uncle': 'aunt', 'aunt': 'uncle',
            'grandfather': 'grandmother', 'grandmother': 'grandfather',
            'grandson': 'granddaughter', 'granddaughter': 'grandson',
            
            # 职业相关（一些有性别特指的）
            'businessman': 'businesswoman', 'businesswoman': 'businessman',
            'policeman': 'policewoman', 'policewoman': 'policeman',
            'fireman': 'firewoman', 'firewoman': 'fireman',
        }
        
        # 非二元性别词汇（保持不变或特殊处理）
        self.neutral_words = {
            'person', 'people', 'individual', 'human', 'adult',
            'parent', 'child', 'sibling', 'spouse', 'partner'
        }
    
    def get_gender_opposite(self, word: str) -> str:
        """获取性别对应词"""
        word_lower = word.lower()
        if word_lower in self.gender_pairs:
            opposite = self.gender_pairs[word_lower]
            # 保持原始大小写
            if word.isupper():
                return opposite.upper()
            elif word.istitle():
                return opposite.title()
            else:
                return opposite
        return word
    
    def extract_gender_words(self, text: str) -> List[str]:
        """提取文本中的性别词汇"""
        words = re.findall(r'\b\w+\b', text.lower())
        gender_words = []
        for word in words:
            if word in self.gender_pairs or word in self.neutral_words:
                gender_words.append(word)
        return gender_words

class StereotypeConverter:
    """刻板印象转换器：将stereotype转换为训练格式"""
    
    def __init__(self):
        self.gender_processor = GenderWordProcessor()
    
    def convert_to_masked_format(self, stereotype: str) -> Dict:
        """将stereotype转换为masked格式"""
        # 检测性别词汇
        gender_words = self.gender_processor.extract_gender_words(stereotype)
        
        if not gender_words:
            return None
        
        # 创建masked版本
        masked_text = stereotype
        gender_positions = []
        
        for word in gender_words:
            if word in self.gender_processor.gender_pairs:
                # 找到词汇位置并替换为[MASK]
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = list(re.finditer(pattern, masked_text, re.IGNORECASE))
                for match in matches:
                    gender_positions.append((match.start(), match.end(), word))
                masked_text = re.sub(pattern, '[MASK]', masked_text, flags=re.IGNORECASE)
        
        return {
            'original': stereotype,
            'masked': masked_text,
            'gender_words': gender_words,
            'gender_positions': gender_positions
        }
    
    def create_gender_pairs(self, masked_data: Dict) -> Tuple[str, str]:
        """基于masked数据创建性别对比对"""
        if not masked_data:
            return None, None
        
        masked_text = masked_data['masked']
        gender_words = masked_data['gender_words']
        
        # 创建男性版本
        male_text = masked_text
        female_text = masked_text
        
        for word in gender_words:
            if word in self.gender_processor.gender_pairs:
                opposite = self.gender_processor.get_gender_opposite(word)
                
                # 替换为对应性别
                if word in ['men', 'man', 'he', 'him', 'his', 'male', 'father', 'son', 'brother']:
                    # 这是男性词汇，保持male_text不变，female_text替换
                    male_text = male_text.replace('[MASK]', word, 1)
                    female_text = female_text.replace('[MASK]', opposite, 1)
                else:
                    # 这是女性词汇，保持female_text不变，male_text替换
                    female_text = female_text.replace('[MASK]', word, 1)
                    male_text = male_text.replace('[MASK]', opposite, 1)
        
        return male_text, female_text

class JobsConverter:
    """职业偏见转换器"""
    
    def __init__(self):
        self.gender_processor = GenderWordProcessor()
    
    def create_job_prompts(self, job_name: str) -> Dict:
        """为职业创建性别对比prompt"""
        base_template = f"这个{job_name}是"
        
        return {
            'base': base_template,
            'male': f"这个{job_name}是他",
            'female': f"这个{job_name}是她",
            'neutral': f"这个{job_name}是一个人",
            'job': job_name
        }

class DebiasDataLoader:
    """去偏见数据加载器"""
    
    def __init__(self):
        self.stereotype_converter = StereotypeConverter()
        self.jobs_converter = JobsConverter()
    
    def load_direct_probe_data(self) -> List[Dict]:
        """加载DirectProbe数据并转换"""
        from genderbench.probes.direct.direct_probe import DirectProbe
        
        probe = DirectProbe()
        items = probe._create_probe_items()
        
        converted_data = []
        for item in items:
            if hasattr(item, 'prompts') and item.prompts:
                # 提取原始stereotype
                prompt_text = item.prompts[0].text
                # 从"Do you agree with the following statement? "XXX". Please answer..."中提取XXX
                match = re.search(r'"([^"]+)"', prompt_text)
                if match:
                    stereotype = match.group(1)
                    
                    # 转换为masked格式
                    masked_data = self.stereotype_converter.convert_to_masked_format(stereotype)
                    if masked_data:
                        male_text, female_text = self.stereotype_converter.create_gender_pairs(masked_data)
                        if male_text and female_text:
                            converted_data.append({
                                'type': 'stereotype',
                                'original': stereotype,
                                'masked': masked_data['masked'],
                                'male_version': male_text,
                                'female_version': female_text,
                                'source': item.metadata.get('source', 'unknown')
                            })
        
        return converted_data
    
    def load_jobs_probe_data(self) -> List[Dict]:
        """加载JobsProbe数据并转换"""
        try:
            from genderbench.probes.jobs_lum.jobs_lum_probe import JobsLumProbe
            
            probe = JobsLumProbe()
            items = probe._create_probe_items()
            
            converted_data = []
            for item in items:
                if hasattr(item, 'prompts') and item.prompts:
                    prompt_text = item.prompts[0].text
                    # 从"I am writing a novel and I need a character who works as a/an XXX"中提取职业
                    match = re.search(r'works as a/an (.+?)\. Please', prompt_text)
                    if match:
                        job = match.group(1)
                        job_prompts = self.jobs_converter.create_job_prompts(job)
                        converted_data.append({
                            'type': 'job',
                            'job': job,
                            'prompts': job_prompts
                        })
            
            return converted_data
        except Exception as e:
            print(f"加载Jobs数据失败: {e}")
            return []

def content_similarity_loss(text1: str, text2: str, tokenizer) -> torch.Tensor:
    """计算两个文本除性别词外的相似度损失"""
    processor = GenderWordProcessor()
    
    # 移除性别词汇
    words1 = text1.split()
    words2 = text2.split()
    
    filtered_words1 = [word for word in words1 
                      if word.lower() not in processor.gender_pairs 
                      and word.lower() not in processor.neutral_words]
    filtered_words2 = [word for word in words2 
                      if word.lower() not in processor.gender_pairs 
                      and word.lower() not in processor.neutral_words]
    
    # 计算编辑距离或余弦相似度
    # 这里简化为长度差异惩罚
    length_diff = abs(len(filtered_words1) - len(filtered_words2))
    
    # 计算词汇重叠度
    set1, set2 = set(filtered_words1), set(filtered_words2)
    overlap = len(set1.intersection(set2))
    total = len(set1.union(set2))
    
    similarity = overlap / total if total > 0 else 0
    consistency_loss = 1.0 - similarity + 0.1 * length_diff
    
    return torch.tensor(consistency_loss, dtype=torch.float32)

def create_training_batch(converted_data: List[Dict], batch_size: int = 4) -> List[Dict]:
    """创建训练批次"""
    import random
    
    # 随机选择数据
    selected = random.sample(converted_data, min(batch_size, len(converted_data)))
    
    batch = []
    for item in selected:
        if item['type'] == 'stereotype':
            batch.append({
                'male_prompt': f"请对这个陈述表态: '{item['male_version']}'",
                'female_prompt': f"请对这个陈述表态: '{item['female_version']}'",
                'bias_type': 'stereotype'
            })
        elif item['type'] == 'job':
            batch.append({
                'male_prompt': f"请描述：{item['prompts']['male']}",
                'female_prompt': f"请描述：{item['prompts']['female']}",
                'bias_type': 'occupation'
            })
    
    return batch

# 示例使用函数
def demonstrate_conversion():
    """演示转换功能"""
    print("🔄 === 数据转换演示 ===")
    
    # 加载数据
    loader = DebiasDataLoader()
    
    # 加载stereotype数据
    stereotype_data = loader.load_direct_probe_data()
    print(f"📊 加载了 {len(stereotype_data)} 个stereotype")
    
    # 显示转换示例
    print("\n🎯 转换示例:")
    for i, item in enumerate(stereotype_data[:3]):
        print(f"  {i+1}. 原始: {item['original']}")
        print(f"     男性版本: {item['male_version']}")
        print(f"     女性版本: {item['female_version']}")
        print()
    
    # 加载职业数据
    jobs_data = loader.load_jobs_probe_data()
    print(f"📊 加载了 {len(jobs_data)} 个职业")
    
    # 显示职业示例
    print("\n💼 职业示例:")
    for i, item in enumerate(jobs_data[:3]):
        print(f"  {i+1}. 职业: {item['job']}")
        print(f"     男性: {item['prompts']['male']}")
        print(f"     女性: {item['prompts']['female']}")
        print()

if __name__ == "__main__":
    demonstrate_conversion() 