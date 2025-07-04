#!/usr/bin/env python3
"""
GenderBench数据探索脚本
分析不同探测器的数据格式，为masked LLM转换做准备
"""

import sys
import os
from pathlib import Path

# 添加genderbench到路径
sys.path.append('./genderbench')

def explore_direct_probe():
    """探索DirectProbe：刻板印象陈述"""
    print("🔍 === DirectProbe探索 ===")
    
    from genderbench.probes.direct.direct_probe import DirectProbe
    
    probe = DirectProbe()
    print(f"📊 Probe类型: {type(probe).__name__}")
    print(f"🎯 模板: {probe.template}")
    
    # 查看stereotype数据
    try:
        items = probe._create_probe_items()
        print(f"📝 总stereotype数量: {len(items)}")
        
        # 显示前几个例子
        print("\n📋 Stereotype样本:")
        for i, item in enumerate(items[:10]):
            if hasattr(item, 'prompts') and item.prompts:
                prompt_text = item.prompts[0].text
                print(f"  {i+1}. {prompt_text}")
        
        # 按来源分析
        sources = {}
        for item in items:
            source = item.metadata.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"\n📈 按来源统计:")
        for source, count in sources.items():
            print(f"  {source}: {count}个")
            
    except Exception as e:
        print(f"❌ 加载DirectProbe数据失败: {e}")

def explore_jobs_probe():
    """探索JobsLumProbe：职业相关偏见"""
    print("\n🔍 === JobsLumProbe探索 ===")
    
    try:
        from genderbench.probes.jobs_lum.jobs_lum_probe import JobsLumProbe
        
        probe = JobsLumProbe()
        print(f"📊 Probe类型: {type(probe).__name__}")
        
        items = probe._create_probe_items()
        print(f"🏢 总职业数量: {len(items)}")
        
        # 显示前几个职业例子
        print("\n💼 职业样本:")
        for i, item in enumerate(items[:10]):
            if hasattr(item, 'prompts') and item.prompts:
                prompt_text = item.prompts[0].text
                print(f"  {i+1}. {prompt_text}")
                
    except Exception as e:
        print(f"❌ 加载JobsLumProbe失败: {e}")

def explore_gest_probe():
    """探索GestProbe：性别刻板印象归属"""
    print("\n🔍 === GestProbe探索 ===")
    
    try:
        from genderbench.probes.gest.gest_probe import GestProbe
        
        probe = GestProbe()
        print(f"📊 Probe类型: {type(probe).__name__}")
        
        items = probe._create_probe_items()
        print(f"🎭 总测试项数量: {len(items)}")
        
        # 显示前几个例子
        print("\n🗣️ GEST样本:")
        for i, item in enumerate(items[:5]):
            if hasattr(item, 'prompts') and item.prompts:
                prompt_text = item.prompts[0].text
                print(f"  {i+1}. {prompt_text}")
                
    except Exception as e:
        print(f"❌ 加载GestProbe失败: {e}")

def explore_resources():
    """探索资源文件，了解原始数据"""
    print("\n🔍 === 资源文件探索 ===")
    
    resources_path = Path("./genderbench/genderbench/resources")
    if resources_path.exists():
        print(f"📁 资源目录: {resources_path}")
        
        # 探索stereotype文件
        sbic_file = resources_path / "sbic_stereotypes" / "stereotypes.txt"
        if sbic_file.exists():
            with open(sbic_file, 'r') as f:
                sbic_lines = f.readlines()
            print(f"📄 SBIC stereotypes: {len(sbic_lines)}行")
            
            print("\n🔸 SBIC样本 (前10个):")
            for i, line in enumerate(sbic_lines[:10]):
                print(f"  {i+1}. {line.strip()}")
        
        gest_file = resources_path / "gest_stereotypes" / "stereotypes.txt"
        if gest_file.exists():
            with open(gest_file, 'r') as f:
                gest_lines = f.readlines()
            print(f"\n📄 GEST stereotypes: {len(gest_lines)}行")
            
            print("\n🔸 GEST样本 (前10个):")
            for i, line in enumerate(gest_lines[:10]):
                print(f"  {i+1}. {line.strip()}")
                
        # 探索其他资源
        print(f"\n📂 所有资源目录:")
        for subdir in resources_path.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*"))
                print(f"  📁 {subdir.name}/: {len(files)}个文件")
                for file in files[:3]:  # 显示前3个文件
                    print(f"    📄 {file.name}")
    
    else:
        print("❌ 资源目录不存在")

def analyze_for_masked_llm():
    """分析数据，为转换为masked LLM格式做准备"""
    print("\n🔍 === Masked LLM转换分析 ===")
    
    # 分析stereotype模式
    resources_path = Path("./genderbench/genderbench/resources")
    sbic_file = resources_path / "sbic_stereotypes" / "stereotypes.txt"
    
    if sbic_file.exists():
        with open(sbic_file, 'r') as f:
            stereotypes = [line.strip() for line in f.readlines()]
        
        # 分析性别词汇模式
        gender_patterns = {
            'men': 0, 'women': 0, 'man': 0, 'woman': 0,
            'male': 0, 'female': 0, 'trans': 0, 'nonbinary': 0,
            'he': 0, 'she': 0, 'his': 0, 'her': 0
        }
        
        for stereotype in stereotypes:
            lower_text = stereotype.lower()
            for pattern in gender_patterns:
                if pattern in lower_text:
                    gender_patterns[pattern] += 1
        
        print("🎯 性别词汇出现频次:")
        for pattern, count in sorted(gender_patterns.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {pattern}: {count}次")
        
        # 找出适合转换的stereotype
        print("\n🔄 适合Masked LLM转换的stereotype样本:")
        convertible = []
        for stereotype in stereotypes[:20]:
            if any(word in stereotype.lower() for word in ['men are', 'women are', 'man is', 'woman is']):
                convertible.append(stereotype)
        
        for i, stereotype in enumerate(convertible[:5]):
            print(f"  原文: {stereotype}")
            # 示例转换
            masked = stereotype.replace('men', '[GENDER]').replace('women', '[GENDER]')
            masked = masked.replace('man', '[GENDER]').replace('woman', '[GENDER]')
            print(f"  转换: {masked}")
            print()

def main():
    """主函数：运行所有探索"""
    print("🚀 GenderBench数据探索开始")
    print("=" * 50)
    
    # 基础信息
    print(f"📍 当前目录: {os.getcwd()}")
    print(f"🐍 Python版本: {sys.version}")
    
    # 检查genderbench是否可用
    try:
        import genderbench
        print(f"✅ GenderBench版本: {genderbench.__version__ if hasattr(genderbench, '__version__') else '已安装'}")
    except ImportError:
        print("❌ GenderBench未安装")
        return
    
    # 探索不同探测器
    explore_direct_probe()
    explore_jobs_probe() 
    explore_gest_probe()
    
    # 探索资源文件
    explore_resources()
    
    # 分析转换可能性
    analyze_for_masked_llm()
    
    print("\n🎉 探索完成！")
    print("💡 建议: 基于以上分析，我们可以设计数据转换和约束生成策略")

if __name__ == "__main__":
    main() 