#!/usr/bin/env python3
"""
Gender Debias Utilities V2 - 改进版本
修复多性别词汇处理bug，添加职业翻译
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
    """性别词汇处理器 - 改进版"""
    
    def __init__(self):
        # 性别词汇映射表
        self.gender_pairs = {
            # 基础代词
            'he': 'she', 'she': 'he',
            'him': 'her', 'her': 'him', 
            'his': 'hers', 'hers': 'his',
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
            'husband': 'wife', 'wife': 'husband',
            
            # 职业相关
            'businessman': 'businesswoman', 'businesswoman': 'businessman',
            'policeman': 'policewoman', 'policewoman': 'policeman',
            'fireman': 'firewoman', 'firewoman': 'fireman',
        }
        
        # 职业名称翻译
        self.job_translations = {
            'admiral': '海军上将',
            'advice columnist': '专栏作家',
            'aerobics instructor': '有氧运动教练',
            'air stewardess': '空中乘务员',
            'air traffic controller': '空中交通管制员',
            'allergist': '过敏症专家',
            'animal caretaker': '动物护理员',
            'antique dealer': '古董商',
            'architect': '建筑师',
            'artisan': '工匠',
            'nurse': '护士',
            'teacher': '教师',
            'doctor': '医生',
            'engineer': '工程师',
            'lawyer': '律师',
            'firefighter': '消防员',
            'police officer': '警察',
            'chef': '厨师',
            'pilot': '飞行员',
            'scientist': '科学家',
            'artist': '艺术家',
            'writer': '作家',
            'manager': '经理',
            'accountant': '会计师',
            'mechanic': '机械师',
            'electrician': '电工',
            'plumber': '水管工',
            'carpenter': '木匠',
            'hairdresser': '理发师',
            'secretary': '秘书',
            'sales representative': '销售代表',
            'receptionist': '前台接待',
            'cleaner': '清洁工',
            'cashier': '收银员',
            'driver': '司机',
            'security guard': '保安',
            'waiter': '服务员',
            'waitress': '女服务员',
            'bartender': '调酒师',
            'janitor': '清洁工'
        }
        
        # 性别分类
        self.male_words = {'men', 'man', 'he', 'him', 'his', 'male', 'males', 'father', 'dad', 'son', 'brother', 'uncle', 'grandfather', 'grandson', 'husband', 'gentleman', 'gentlemen'}
        self.female_words = {'women', 'woman', 'she', 'her', 'hers', 'female', 'females', 'mother', 'mom', 'daughter', 'sister', 'aunt', 'grandmother', 'granddaughter', 'wife', 'lady', 'ladies'}
    
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
    
    def translate_job(self, job: str) -> str:
        """翻译职业名称"""
        return self.job_translations.get(job.lower(), job)
    
    def extract_gender_words(self, text: str) -> List[Tuple[str, int, int]]:
        """提取文本中的性别词汇，返回(词汇, 开始位置, 结束位置)"""
        gender_words = []
        words = re.finditer(r'\b\w+\b', text)
        
        for match in words:
            word = match.group().lower()
            if word in self.gender_pairs:
                gender_words.append((word, match.start(), match.end()))
        
        return gender_words

class SmartStereotypeConverter:
    """智能刻板印象转换器 - 改进版"""
    
    def __init__(self):
        self.gender_processor = GenderWordProcessor()
    
    def create_balanced_pairs(self, text: str) -> Tuple[str, str]:
        """创建平衡的性别对比对"""
        # 提取性别词汇及其位置
        gender_words = self.gender_processor.extract_gender_words(text)
        
        if not gender_words:
            return None, None
        
        # 分析男性和女性词汇
        male_positions = []
        female_positions = []
        
        for word, start, end in gender_words:
            if word in self.gender_processor.male_words:
                male_positions.append((word, start, end))
            elif word in self.gender_processor.female_words:
                female_positions.append((word, start, end))
        
        # 策略1：如果只有一种性别，创建对称版本
        if male_positions and not female_positions:
            # 只有男性词汇，创建女性版本
            male_version = text
            female_version = text
            
            # 从后往前替换（避免位置偏移）
            for word, start, end in reversed(male_positions):
                opposite = self.gender_processor.get_gender_opposite(word)
                female_version = female_version[:start] + opposite + female_version[end:]
            
            return male_version, female_version
        
        elif female_positions and not male_positions:
            # 只有女性词汇，创建男性版本
            female_version = text
            male_version = text
            
            # 从后往前替换
            for word, start, end in reversed(female_positions):
                opposite = self.gender_processor.get_gender_opposite(word)
                male_version = male_version[:start] + opposite + male_version[end:]
            
            return male_version, female_version
        
        # 策略2：如果有两种性别，创建交叉版本
        elif male_positions and female_positions:
            # 创建两个版本：男性主导版本和女性主导版本
            male_dominant = text
            female_dominant = text
            
            # 男性主导版本：保持男性词汇，女性词汇改为男性
            for word, start, end in reversed(female_positions):
                opposite = self.gender_processor.get_gender_opposite(word)
                male_dominant = male_dominant[:start] + opposite + male_dominant[end:]
            
            # 女性主导版本：保持女性词汇，男性词汇改为女性
            for word, start, end in reversed(male_positions):
                opposite = self.gender_processor.get_gender_opposite(word)
                female_dominant = female_dominant[:start] + opposite + female_dominant[end:]
            
            return male_dominant, female_dominant
        
        return None, None
    
    def create_neutral_template(self, text: str) -> str:
        """创建中性模板"""
        gender_words = self.gender_processor.extract_gender_words(text)
        
        if not gender_words:
            return text
        
        neutral_text = text
        # 从后往前替换为[GENDER]
        for word, start, end in reversed(gender_words):
            neutral_text = neutral_text[:start] + '[GENDER]' + neutral_text[end:]
        
        return neutral_text

class ImprovedDebiasDataLoader:
    """改进的去偏见数据加载器"""
    
    def __init__(self):
        self.stereotype_converter = SmartStereotypeConverter()
        self.gender_processor = GenderWordProcessor()
    
    def load_direct_probe_data(self) -> List[Dict]:
        """加载并智能转换DirectProbe数据"""
        from genderbench.probes.direct.direct_probe import DirectProbe
        
        probe = DirectProbe()
        items = probe._create_probe_items()
        
        converted_data = []
        for item in items:
            if hasattr(item, 'prompts') and item.prompts:
                # 提取原始stereotype
                prompt_text = item.prompts[0].text
                match = re.search(r'"([^"]+)"', prompt_text)
                if match:
                    stereotype = match.group(1)
                    
                    # 智能转换
                    male_version, female_version = self.stereotype_converter.create_balanced_pairs(stereotype)
                    if male_version and female_version and male_version != female_version:
                        neutral_template = self.stereotype_converter.create_neutral_template(stereotype)
                        
                        converted_data.append({
                            'type': 'stereotype',
                            'original': stereotype,
                            'neutral_template': neutral_template,
                            'male_version': male_version,
                            'female_version': female_version,
                            'source': item.metadata.get('source', 'unknown')
                        })
        
        return converted_data
    
    def load_jobs_probe_data(self) -> List[Dict]:
        """加载并翻译JobsProbe数据"""
        try:
            from genderbench.probes.jobs_lum.jobs_lum_probe import JobsLumProbe
            
            probe = JobsLumProbe()
            items = probe._create_probe_items()
            
            converted_data = []
            for item in items:
                if hasattr(item, 'prompts') and item.prompts:
                    prompt_text = item.prompts[0].text
                    match = re.search(r'works as a/an (.+?)\. Please', prompt_text)
                    if match:
                        job_en = match.group(1)
                        job_cn = self.gender_processor.translate_job(job_en)
                        
                        converted_data.append({
                            'type': 'job',
                            'job_en': job_en,
                            'job_cn': job_cn,
                            'prompts': {
                                'base': f"这个{job_cn}是",
                                'male': f"这个{job_cn}是他",
                                'female': f"这个{job_cn}是她",
                                'neutral': f"这个{job_cn}是一个专业人士"
                            }
                        })
            
            return converted_data
        except Exception as e:
            print(f"加载Jobs数据失败: {e}")
            return []

def analyze_conversion_quality(converted_data: List[Dict]) -> Dict:
    """分析转换质量"""
    analysis = {
        'total_items': len(converted_data),
        'valid_conversions': 0,
        'identical_pairs': 0,
        'word_difference_stats': [],
        'examples': []
    }
    
    for item in converted_data:
        if item['type'] == 'stereotype':
            male_words = set(item['male_version'].lower().split())
            female_words = set(item['female_version'].lower().split())
            
            # 计算词汇差异
            diff_count = len(male_words.symmetric_difference(female_words))
            analysis['word_difference_stats'].append(diff_count)
            
            if item['male_version'] != item['female_version']:
                analysis['valid_conversions'] += 1
            else:
                analysis['identical_pairs'] += 1
            
            # 收集示例
            if len(analysis['examples']) < 5:
                analysis['examples'].append({
                    'original': item['original'],
                    'male': item['male_version'],
                    'female': item['female_version'],
                    'neutral': item['neutral_template']
                })
    
    return analysis

def demonstrate_improved_conversion():
    """演示改进的转换功能"""
    print("🚀 === 改进版数据转换演示 ===")
    
    # 加载数据
    loader = ImprovedDebiasDataLoader()
    
    # 加载stereotype数据
    print("📊 加载stereotype数据...")
    stereotype_data = loader.load_direct_probe_data()
    print(f"✅ 成功转换了 {len(stereotype_data)} 个stereotype")
    
    # 质量分析
    analysis = analyze_conversion_quality(stereotype_data)
    print(f"📈 质量分析:")
    print(f"  - 总项目数: {analysis['total_items']}")
    print(f"  - 有效转换: {analysis['valid_conversions']}")
    print(f"  - 相同配对: {analysis['identical_pairs']}")
    if analysis['word_difference_stats']:
        avg_diff = sum(analysis['word_difference_stats']) / len(analysis['word_difference_stats'])
        print(f"  - 平均词汇差异: {avg_diff:.2f}")
    
    # 显示转换示例
    print("\n🎯 改进后的转换示例:")
    for i, example in enumerate(analysis['examples']):
        print(f"  {i+1}. 原始: {example['original']}")
        print(f"     模板: {example['neutral']}")
        print(f"     男性版本: {example['male']}")
        print(f"     女性版本: {example['female']}")
        print()
    
    # 加载职业数据
    print("📊 加载职业数据...")
    jobs_data = loader.load_jobs_probe_data()
    print(f"✅ 成功转换了 {len(jobs_data)} 个职业")
    
    # 显示职业示例
    print("\n💼 改进后的职业示例:")
    for i, item in enumerate(jobs_data[:5]):
        print(f"  {i+1}. 职业: {item['job_en']} ({item['job_cn']})")
        print(f"     男性: {item['prompts']['male']}")
        print(f"     女性: {item['prompts']['female']}")
        print(f"     中性: {item['prompts']['neutral']}")
        print()

if __name__ == "__main__":
    demonstrate_improved_conversion() 