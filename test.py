# simple_direct_run.py
# The simplest possible version - no workflow, just direct execution

import sys
import os
sys.path.append(os.path.dirname(__file__))

from dspy_module import generate_thread_from_topic, configure_llm

def main():
    """Main execution function"""
    
    # Initialize DSPy LLM
    print("🚀 Initializing Gemini LLM...")
    configure_llm()
    
    # Define the task
    topic = "The hidden benefits of walking 10,000 steps a day"
    template = "Lessons learned"
    
    print(f"\n📝 Topic: {topic}")
    print(f"📋 Template: {template}\n")
    print("="*80)
    print("🎨 Generating thread...\n")
    
    # Call the DSPy function directly
    result = generate_thread_from_topic(topic=topic, template=template)
    
    # Display results
    print("\n" + "="*80)
    print("🎉 GENERATION COMPLETE! FINAL OUTPUT:")
    print("="*80 + "\n")
    
    if isinstance(result, dict):
        # Display hook
        print("🎣 HOOK:")
        print("-" * 80)
        print(result.get('hook', 'N/A'))
        print()
        
        # Display tweets
        print("📱 THREAD:")
        print("-" * 80)
        tweets = result.get('tweets', [])
        for i, tweet in enumerate(tweets, 1):
            print(f"{i}/ {tweet}\n")
        
        # Display TL;DR if available
        if 'tldr' in result:
            print("-" * 80)
            print("📌 TL;DR:")
            print(result['tldr'])
            print()
        
        # Display CTA if available
        if 'cta' in result:
            print("-" * 80)
            print("🎯 CALL TO ACTION:")
            print(result['cta'])
            print()
        
        print("="*80)
        print(f"✅ Generated {len(tweets)} tweets successfully!")
        
    else:
        print("⚠️ Unexpected result format:")
        print(result)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()