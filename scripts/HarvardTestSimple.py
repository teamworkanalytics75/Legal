import os
import sys
from pathlib import Path

# Set up environment
os.environ['OPENAI_API_KEY'] = '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'

# Add project root to path
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from writer_agents.code.demo_langchain_workflow import LangChainWorkflow
    
    print('Testing Harvard Schedule 4 Xi Slide Query')
    print('=' * 60)
    
    # Initialize workflow
    workflow = LangChainWorkflow()
    
    # The specific query
    query = '''
    What arguments did Harvard have constructive knowledge of regarding the Schedule 4 Xi slide on or before April 19, 2019?
    
    Please provide a comprehensive analysis including:
    1. Evidence of Harvard awareness of the Schedule 4 Xi slide
    2. Timeline evidence supporting knowledge before April 19, 2019
    3. Legal arguments supporting constructive knowledge claims
    4. Specific documents or communications demonstrating this knowledge
    5. Relevant case law or precedents that support these arguments
    6. Analysis of the legal standard for constructive knowledge
    '''
    
    print('Running query...')
    print(f'Query: {query[:100]}...')
    print()
    
    # Run the query
    result = workflow.run_query(query)
    
    print('Query Results:')
    print('=' * 80)
    print(result)
    print('=' * 80)
    
    # Analyze the response
    print('\nResponse Analysis:')
    
    # Key elements to look for
    key_elements = {
        "constructive knowledge": "constructive knowledge" in result.lower(),
        "timeline analysis": "april 19, 2019" in result.lower() or "2019" in result.lower(),
        "schedule 4 xi slide": "schedule 4" in result.lower() and "xi slide" in result.lower(),
        "harvard awareness": "harvard" in result.lower() and ("aware" in result.lower() or "knowledge" in result.lower()),
        "legal arguments": "legal" in result.lower() and "argument" in result.lower(),
        "evidence": "evidence" in result.lower() or "document" in result.lower(),
        "case law": "case law" in result.lower() or "precedent" in result.lower(),
        "structured analysis": result.count("1.") >= 2 and result.count("2.") >= 1
    }
    
    print("Key Elements Found:")
    for element, found in key_elements.items():
        status = "YES" if found else "NO"
        print(f"  {status} {element}")
    
    # Overall quality score
    score = sum(key_elements.values()) / len(key_elements) * 100
    print(f"\nOverall Quality Score: {score:.1f}%")
    
    if score >= 80:
        print("EXCELLENT - LangChain demonstrated sophisticated legal reasoning")
    elif score >= 60:
        print("GOOD - LangChain showed solid legal analysis capabilities")
    elif score >= 40:
        print("FAIR - LangChain provided basic legal information")
    else:
        print("POOR - LangChain struggled with complex legal reasoning")
    
except Exception as e:
    print(f'Query failed: {e}')
    import traceback
    traceback.print_exc()
