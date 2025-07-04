#!/usr/bin/env python3
"""
Gender Debias Utilities - English Version
Pure English implementation for gender debiasing
Core idea: Generate identical responses except for gender
"""

import sys
import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add genderbench to path
sys.path.append('./genderbench')

class EnglishGenderProcessor:
    """English Gender Word Processor"""
    
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
            'guy': 'gal', 'gal': 'guy',
            'guys': 'gals', 'gals': 'guys',
        }
        
        # Gender classification
        self.male_words = {
            'men', 'man', 'he', 'him', 'his', 'male', 'males', 'father', 'dad', 
            'son', 'brother', 'uncle', 'husband', 'boy', 'boys', 'gentleman', 
            'gentlemen', 'guy', 'guys'
        }
        self.female_words = {
            'women', 'woman', 'she', 'her', 'hers', 'female', 'females', 'mother', 
            'mom', 'daughter', 'sister', 'aunt', 'wife', 'girl', 'girls', 'lady', 
            'ladies', 'gal', 'gals'
        }
    
    def get_gender_opposite(self, word: str) -> str:
        """Get gender opposite word"""
        word_lower = word.lower()
        if word_lower in self.gender_pairs:
            opposite = self.gender_pairs[word_lower]
            # Preserve original case
            if word.isupper():
                return opposite.upper()
            elif word.istitle():
                return opposite.title()
            else:
                return opposite
        return word
    
    def extract_gender_words_with_roles(self, text: str) -> List[Dict]:
        """Extract gender words with their roles in sentence"""
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
        """Analyze word role in sentence (subject, object, modifier)"""
        before_text = text[:start].strip()
        after_text = text[end:].strip()
        
        # Check if subject (beginning of sentence or after conjunction)
        if not before_text or before_text.endswith('.') or before_text.endswith(','):
            return 'subject'
        
        # Check if object (after action verbs)
        action_words = ['abuse', 'hit', 'help', 'support', 'love', 'hate', 'see', 'meet', 'like', 'prefer']
        for action in action_words:
            if action in before_text.lower().split()[-3:]:
                return 'object'
        
        # Default to modifier
        return 'modifier'

class EnglishStereotypeConverter:
    """English Stereotype Converter"""
    
    def __init__(self):
        self.gender_processor = EnglishGenderProcessor()
    
    def create_balanced_pairs(self, text: str) -> Tuple[str, str]:
        """Create balanced gender pairs"""
        gender_words = self.gender_processor.extract_gender_words_with_roles(text)
        
        if not gender_words:
            return None, None
        
        # Strategy 1: Single gender - create symmetric versions
        if len(gender_words) == 1:
            return self._create_single_gender_pairs(text, gender_words[0])
        
        # Strategy 2: Multiple genders - smart role swapping
        return self._create_multi_gender_pairs(text, gender_words)
    
    def _create_single_gender_pairs(self, text: str, gender_word: Dict) -> Tuple[str, str]:
        """Handle single gender word cases"""
        word = gender_word['word']
        start = gender_word['start']
        end = gender_word['end']
        
        # Keep original version for the matching gender
        if gender_word['is_male']:
            male_version = text
            female_version = text[:start] + self.gender_processor.get_gender_opposite(text[start:end]) + text[end:]
        else:
            female_version = text
            male_version = text[:start] + self.gender_processor.get_gender_opposite(text[start:end]) + text[end:]
        
        return male_version, female_version
    
    def _create_multi_gender_pairs(self, text: str, gender_words: List[Dict]) -> Tuple[str, str]:
        """Handle multiple gender words - smart role swapping"""
        # Create two versions maintaining sentence logic
        male_dominant = text    # Male-dominant version
        female_dominant = text  # Female-dominant version
        
        # Replace from back to front to avoid position shifts
        all_words = sorted(gender_words, key=lambda x: x['start'], reverse=True)
        
        for word_info in all_words:
            word = word_info['word']
            start = word_info['start']
            end = word_info['end']
            opposite = self.gender_processor.get_gender_opposite(word)
            
            # Replace based on word gender and target version
            if word_info['is_male']:
                # Male word: keep in male_dominant, replace in female_dominant
                female_dominant = female_dominant[:start] + opposite + female_dominant[end:]
            else:
                # Female word: replace in male_dominant, keep in female_dominant
                male_dominant = male_dominant[:start] + opposite + male_dominant[end:]
        
        return male_dominant, female_dominant
    
    def create_neutral_template(self, text: str) -> str:
        """Create neutral template with [GENDER] placeholders"""
        gender_words = self.gender_processor.extract_gender_words_with_roles(text)
        
        if not gender_words:
            return text
        
        neutral_text = text
        # Replace from back to front with [GENDER]
        for word_info in reversed(gender_words):
            start = word_info['start']
            end = word_info['end']
            neutral_text = neutral_text[:start] + '[GENDER]' + neutral_text[end:]
        
        return neutral_text

class EnglishDebiasDataLoader:
    """English Debias Data Loader"""
    
    def __init__(self):
        self.converter = EnglishStereotypeConverter()
        self.gender_processor = EnglishGenderProcessor()
    
    def load_direct_probe_data(self) -> List[Dict]:
        """Load and convert DirectProbe data"""
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
                    
                    # Smart conversion
                    version1, version2 = self.converter.create_balanced_pairs(stereotype)
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
        """Load and process JobsProbe data in English"""
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
                        job = match.group(1)
                        
                        converted_data.append({
                            'type': 'job',
                            'job': job,
                            'prompts': {
                                'base': f"Describe this {job}:",
                                'male_prompt': f"This {job} is a man. Describe his characteristics:",
                                'female_prompt': f"This {job} is a woman. Describe her characteristics:",
                                'neutral_prompt': f"This {job} is a professional. Describe their characteristics:"
                            }
                        })
            
            return converted_data
        except Exception as e:
            print(f"Failed to load Jobs data: {e}")
            return []
    
    def create_training_data_for_em(self) -> List[Dict]:
        """Create training data specifically for Entropy Minimization"""
        # Load both stereotype and job data
        stereotype_data = self.load_direct_probe_data()
        job_data = self.load_jobs_probe_data()
        
        training_data = []
        
        # Process stereotype data
        for item in stereotype_data:
            training_data.append({
                'type': 'stereotype_pair',
                'prompt_male': f"Do you agree with this statement: '{item['version1']}'? Please explain your reasoning.",
                'prompt_female': f"Do you agree with this statement: '{item['version2']}'? Please explain your reasoning.",
                'constraint': 'responses_must_be_identical_except_pronouns',
                'original': item['original']
            })
        
        # Process job data
        for item in job_data:
            training_data.append({
                'type': 'job_pair',
                'prompt_male': item['prompts']['male_prompt'],
                'prompt_female': item['prompts']['female_prompt'],
                'constraint': 'responses_must_be_identical_except_pronouns',
                'job': item['job']
            })
        
        return training_data

def demonstrate_english_conversion():
    """Demonstrate English conversion functionality"""
    print("🚀 === English Gender Debias Conversion Demo ===")
    
    # Load data
    loader = EnglishDebiasDataLoader()
    
    # Load stereotype data
    print("📊 Loading stereotype data...")
    stereotype_data = loader.load_direct_probe_data()
    print(f"✅ Successfully converted {len(stereotype_data)} stereotypes")
    
    # Show conversion examples
    print("\n🎯 English Conversion Examples:")
    for i, item in enumerate(stereotype_data[:8]):
        print(f"  {i+1}. Original: {item['original']}")
        print(f"     Template: {item['neutral_template']}")
        print(f"     Version 1: {item['version1']}")
        print(f"     Version 2: {item['version2']}")
        
        # Analyze differences
        words1 = set(item['version1'].lower().split())
        words2 = set(item['version2'].lower().split())
        diff = words1.symmetric_difference(words2)
        print(f"     Different words: {diff}")
        print()
    
    # Load job data
    print("📊 Loading job data...")
    job_data = loader.load_jobs_probe_data()
    print(f"✅ Successfully loaded {len(job_data)} jobs")
    
    # Show job examples
    print("\n💼 English Job Examples:")
    for i, item in enumerate(job_data[:5]):
        print(f"  {i+1}. Job: {item['job']}")
        print(f"     Base: {item['prompts']['base']}")
        print(f"     Male: {item['prompts']['male_prompt']}")
        print(f"     Female: {item['prompts']['female_prompt']}")
        print(f"     Neutral: {item['prompts']['neutral_prompt']}")
        print()
    
    # Create training data
    print("📊 Creating training data for EM...")
    training_data = loader.create_training_data_for_em()
    print(f"✅ Created {len(training_data)} training pairs")
    
    # Show training data examples
    print("\n🎯 Training Data Examples:")
    for i, item in enumerate(training_data[:3]):
        print(f"  {i+1}. Type: {item['type']}")
        print(f"     Male prompt: {item['prompt_male']}")
        print(f"     Female prompt: {item['prompt_female']}")
        print(f"     Constraint: {item['constraint']}")
        print()
    
    # Analysis
    print("📊 Analysis:")
    stereotype_pairs = sum(1 for item in training_data if item['type'] == 'stereotype_pair')
    job_pairs = sum(1 for item in training_data if item['type'] == 'job_pair')
    print(f"  - Stereotype pairs: {stereotype_pairs}")
    print(f"  - Job pairs: {job_pairs}")
    print(f"  - Total pairs: {len(training_data)}")
    print(f"  - Ready for EM training: ✅")

if __name__ == "__main__":
    demonstrate_english_conversion() 