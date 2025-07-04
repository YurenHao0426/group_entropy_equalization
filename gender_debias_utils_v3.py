#!/usr/bin/env python3
"""
Gender Debias Utilities V3 - 终极版本
真正实现"保持句子结构，只调换性别角色"的智能转换
核心思想：保持句子逻辑完整性，只调换主语和宾语的性别地位
"""

import sys
import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# 添加genderbench到路径
sys.path.append('./genderbench')

class AdvancedGenderProcessor:
    """高级性别词汇处理器"""
    
    def __init__(self):
        self.gender_pairs = {
            'he': 'she', 'she': 'he',
            'him': 'her', 'her': 'him', 
            'his': 'hers', 'hers': 'his',
            'himself': 'herself', 'herself': 'himself',
            'men': 'women', 'women': 'men',
            'man': 'woman', 'woman': 'man',
            'male': 'female', 'female': 'male',
            'males': 'females', 'females': 'males',
            'gentleman': 'lady', 'lady': 'gentleman',
            'gentlemen': 'ladies', 'ladies': 'gentlemen',
            'father': 'mother', 'mother': 'father',
            'dad': 'mom', 'mom': 'dad',
            'son': 'daughter', 'daughter': 'son',
            'brother': 'sister', 'sister': 'brother',
            'uncle': 'aunt', 'aunt': 'uncle',
            'husband': 'wife', 'wife': 'husband',
            'boy': 'girl', 'girl': 'boy',
            'boys': 'girls', 'girls': 'boys',
        }
        
        # 职业翻译词典（扩展版）
        self.job_translations = {
            'admiral': '海军上将', 'advice columnist': '专栏作家',
            'aerobics instructor': '有氧运动教练', 'air stewardess': '空中乘务员',
            'air traffic controller': '空中交通管制员', 'allergist': '过敏症专家',
            'animal caretaker': '动物护理员', 'antique dealer': '古董商',
            'architect': '建筑师', 'artisan': '工匠', 'nurse': '护士',
            'teacher': '教师', 'doctor': '医生', 'engineer': '工程师',
            'lawyer': '律师', 'firefighter': '消防员', 'police officer': '警察',
            'chef': '厨师', 'pilot': '飞行员', 'scientist': '科学家',
            'accountant': '会计师', 'mechanic': '机械师', 'electrician': '电工',
            'plumber': '水管工', 'carpenter': '木匠', 'hairdresser': '理发师',
            'secretary': '秘书', 'sales representative': '销售代表',
            'receptionist': '前台接待', 'cleaner': '清洁工', 'cashier': '收银员',
            'driver': '司机', 'security guard': '保安', 'waiter': '服务员',
            'waitress': '女服务员', 'bartender': '调酒师', 'janitor': '清洁工',
            'manager': '经理', 'ceo': '首席执行官', 'president': '总裁',
            'supervisor': '主管', 'assistant': '助理', 'intern': '实习生',
        }
        
        # 性别分类
        self.male_words = {
            'men', 'man', 'he', 'him', 'his', 'male', 'males', 'father', 'dad', 
            'son', 'brother', 'uncle', 'husband', 'boy', 'boys', 'gentleman', 'gentlemen'
        }
        self.female_words = {
            'women', 'woman', 'she', 'her', 'hers', 'female', 'females', 'mother', 'mom', 
            'daughter', 'sister', 'aunt', 'wife', 'girl', 'girls', 'lady', 'ladies'
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
    
    def translate_job(self, job: str) -> str:
        """翻译职业名称"""
        return self.job_translations.get(job.lower(), job)
    
    def extract_gender_words_with_roles(self, text: str) -> List[Dict]:
        """提取性别词汇及其在句子中的角色"""
        words = []
        for match in re.finditer(r'\b\w+\b', text):
            word = match.group().lower()
            if word in self.gender_pairs:
                role = self._analyze_word_role(text, match.start(), match.end(), word)
                words.append({
                    'word': word,
                    'original': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'role': role,
                    'is_male': word in self.male_words,
                    'is_female': word in self.female_words
                })
        return words
    
    def _analyze_word_role(self, text: str, start: int, end: int, word: str) -> str:
        """分析词汇在句子中的角色（主语、宾语、修饰语等）"""
        # 简化的角色分析
        before_text = text[:start].strip()
        after_text = text[end:].strip()
        
        # 判断是否为主语（句子开头或连词后）
        if not before_text or before_text.endswith('.') or before_text.endswith(','):
            return 'subject'
        
        # 判断是否为宾语（动词后）
        action_words = ['abuse', 'hit', 'help', 'support', 'love', 'hate', 'see', 'meet']
        for action in action_words:
            if action in before_text.lower().split()[-3:]:
                return 'object'
        
        # 默认为修饰语
        return 'modifier'

class IntelligentStereotypeConverter:
    """智能刻板印象转换器 - 终极版"""
    
    def __init__(self):
        self.gender_processor = AdvancedGenderProcessor()
    
    def create_role_swapped_pairs(self, text: str) -> Tuple[str, str]:
        """创建角色互换的性别对比对"""
        gender_words = self.gender_processor.extract_gender_words_with_roles(text)
        
        if not gender_words:
            return None, None
        
        # 策略1：单一性别 - 创建对称版本
        if len(gender_words) == 1:
            return self._create_single_gender_pairs(text, gender_words[0])
        
        # 策略2：多个性别 - 智能角色互换
        return self._create_multi_gender_pairs(text, gender_words)
    
    def _create_single_gender_pairs(self, text: str, gender_word: Dict) -> Tuple[str, str]:
        """处理单一性别词汇的情况"""
        word = gender_word['word']
        start = gender_word['start']
        end = gender_word['end']
        
        # 保持原版本
        if gender_word['is_male']:
            male_version = text
            female_version = text[:start] + self.gender_processor.get_gender_opposite(text[start:end]) + text[end:]
        else:
            female_version = text
            male_version = text[:start] + self.gender_processor.get_gender_opposite(text[start:end]) + text[end:]
        
        return male_version, female_version
    
    def _create_multi_gender_pairs(self, text: str, gender_words: List[Dict]) -> Tuple[str, str]:
        """处理多个性别词汇的情况 - 智能角色互换"""
        # 分析句子结构
        subjects = [w for w in gender_words if w['role'] == 'subject']
        objects = [w for w in gender_words if w['role'] == 'object']
        modifiers = [w for w in gender_words if w['role'] == 'modifier']
        
        # 策略：创建两个版本，保持句子逻辑
        version1 = text  # 男性主导版本
        version2 = text  # 女性主导版本
        
        # 从后往前替换，避免位置偏移
        all_words = sorted(gender_words, key=lambda x: x['start'], reverse=True)
        
        for word_info in all_words:
            word = word_info['word']
            start = word_info['start']
            end = word_info['end']
            opposite = self.gender_processor.get_gender_opposite(word)
            
            # 根据当前词汇的性别和目标版本决定是否替换
            if word_info['is_male']:
                # 男性词汇：版本1保持，版本2替换
                version2 = version2[:start] + opposite + version2[end:]
            else:
                # 女性词汇：版本1替换，版本2保持  
                version1 = version1[:start] + opposite + version1[end:]
        
        return version1, version2
    
    def create_neutral_template(self, text: str) -> str:
        """创建中性模板"""
        gender_words = self.gender_processor.extract_gender_words_with_roles(text)
        
        if not gender_words:
            return text
        
        neutral_text = text
        # 从后往前替换为[GENDER]
        for word_info in reversed(gender_words):
            start = word_info['start']
            end = word_info['end']
            neutral_text = neutral_text[:start] + '[GENDER]' + neutral_text[end:]
        
        return neutral_text

class FinalDebiasDataLoader:
    """最终版去偏见数据加载器"""
    
    def __init__(self):
        self.converter = IntelligentStereotypeConverter()
        self.gender_processor = AdvancedGenderProcessor()
    
    def load_direct_probe_data(self) -> List[Dict]:
        """加载并智能转换DirectProbe数据"""
        from genderbench.probes.direct.direct_probe import DirectProbe
        
        probe = DirectProbe()
        items = probe._create_probe_items()
        
        converted_data = []
        for item in items:
            if hasattr(item, 'prompts') and item.prompts:
                prompt_text = item.prompts[0].text
                match = re.search(r'"([^"]+)"', prompt_text)
                if match:
                    stereotype = match.group(1)
                    
                    # 智能转换
                    version1, version2 = self.converter.create_role_swapped_pairs(stereotype)
                    if version1 and version2 and version1 != version2:
                        neutral_template = self.converter.create_neutral_template(stereotype)
                        
                        converted_data.append({
                            'type': 'stereotype',
                            'original': stereotype,
                            'neutral_template': neutral_template,
                            'version1': version1,
                            'version2': version2,
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
                                'base': f"描述一下这个{job_cn}：",
                                'male_prompt': f"这个{job_cn}是男性，请描述他的特点：",
                                'female_prompt': f"这个{job_cn}是女性，请描述她的特点：",
                                'neutral_prompt': f"这个{job_cn}是一个专业人士，请描述其特点："
                            }
                        })
            
            return converted_data
        except Exception as e:
            print(f"加载Jobs数据失败: {e}")
            return []

def demonstrate_final_conversion():
    """演示最终版转换功能"""
    print("🎯 === 最终版智能转换演示 ===")
    
    # 加载数据
    loader = FinalDebiasDataLoader()
    
    # 加载stereotype数据
    print("📊 加载stereotype数据...")
    stereotype_data = loader.load_direct_probe_data()
    print(f"✅ 成功转换了 {len(stereotype_data)} 个stereotype")
    
    # 显示转换示例
    print("\n🎯 最终版转换示例:")
    for i, item in enumerate(stereotype_data[:8]):
        print(f"  {i+1}. 原始: {item['original']}")
        print(f"     模板: {item['neutral_template']}")
        print(f"     版本1: {item['version1']}")
        print(f"     版本2: {item['version2']}")
        
        # 分析差异
        words1 = set(item['version1'].lower().split())
        words2 = set(item['version2'].lower().split())
        diff = words1.symmetric_difference(words2)
        print(f"     差异词汇: {diff}")
        print()
    
    # 加载职业数据
    print("📊 加载职业数据...")
    jobs_data = loader.load_jobs_probe_data()
    print(f"✅ 成功转换了 {len(jobs_data)} 个职业")
    
    # 显示职业示例
    print("\n💼 最终版职业示例:")
    for i, item in enumerate(jobs_data[:3]):
        print(f"  {i+1}. 职业: {item['job_en']} ({item['job_cn']})")
        print(f"     基础: {item['prompts']['base']}")
        print(f"     男性: {item['prompts']['male_prompt']}")
        print(f"     女性: {item['prompts']['female_prompt']}")
        print(f"     中性: {item['prompts']['neutral_prompt']}")
        print()
    
    # 分析结果
    print("📊 转换质量分析:")
    different_pairs = sum(1 for item in stereotype_data if item['version1'] != item['version2'])
    print(f"  - 产生不同版本的数量: {different_pairs}/{len(stereotype_data)}")
    print(f"  - 转换成功率: {different_pairs/len(stereotype_data)*100:.1f}%")

if __name__ == "__main__":
    demonstrate_final_conversion() 