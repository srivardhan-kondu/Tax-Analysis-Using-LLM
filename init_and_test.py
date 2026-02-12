"""
Initialization and test script for Tax Evasion Detection System
Run this to verify your setup is correct
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError:
        print("❌ pandas - run: pip install pandas")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit")
    except ImportError:
        print("❌ streamlit - run: pip install streamlit")
        return False
    
    try:
        from openai import OpenAI
        print("✅ openai")
    except ImportError:
        print("❌ openai - run: pip install openai")
        return False
    
    try:
        import plotly
        print("✅ plotly")
    except ImportError:
        print("❌ plotly - run: pip install plotly")
        return False
    
    return True


def test_project_structure():
    """Test if project structure is correct"""
    print("\n📁 Testing project structure...")
    
    required_dirs = ['data', 'ai', 'engine', 'ui', 'utils', 'tests']
    required_files = [
        'app.py',
        'config.py',
        'requirements.txt',
        '.env',
        'data/data_processor.py',
        'data/data_validator.py',
        'ai/gpt4_analyzer.py',
        'ai/prompts.py',
        'ai/validation.py',
        'engine/risk_predictor.py',
        'ui/authentication.py',
        'ui/data_upload.py',
        'ui/dashboard.py'
    ]
    
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - directory missing")
            all_exist = False
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - file missing")
            all_exist = False
    
    return all_exist


def test_config():
    """Test configuration"""
    print("\n⚙️ Testing configuration...")
    
    try:
        import config
        
        if config.OPENAI_API_KEY:
            if config.OPENAI_API_KEY == "your_openai_api_key_here":
                print("⚠️ OPENAI_API_KEY is not configured (using placeholder)")
                print("   Please update .env file with your actual API key")
            else:
                print("✅ OPENAI_API_KEY is configured")
        else:
            print("❌ OPENAI_API_KEY is missing")
            return False
        
        print(f"✅ Model: {config.OPENAI_MODEL}")
        print(f"✅ Temperature: {config.GPT_TEMPERATURE}")
        print(f"✅ Confidence Threshold: {config.GPT_CONFIDENCE_THRESHOLD}")
        
        return True
    
    except Exception as e:
        print(f"❌ Config error: {str(e)}")
        return False


def test_sample_data():
    """Test if sample data exists"""
    print("\n📊 Testing sample data...")
    
    sample_file = project_root / "data" / "sample_companies.csv"
    
    if sample_file.exists():
        print(f"✅ Sample data found: {sample_file}")
        
        try:
            import pandas as pd
            df = pd.read_csv(sample_file)
            print(f"✅ Sample data loaded: {len(df)} companies")
            return True
        except Exception as e:
            print(f"❌ Error loading sample data: {str(e)}")
            return False
    else:
        print(f"❌ Sample data not found: {sample_file}")
        return False


def test_data_processor():
    """Test data processor module"""
    print("\n🔧 Testing data processor...")
    
    try:
        from data.data_processor import DataProcessor
        import pandas as pd
        
        processor = DataProcessor()
        
        # Create test data
        test_data = pd.DataFrame({
            'company_name': ['Test Company'],
            'sales': [1000000],
            'revenue_growth': [0.15],
            'profit_margin': [0.20],
            'employee_growth': [0.10],
            'debt_ratio': [0.50],
            'operating_expenses': [700000],
            'tax_to_revenue_ratio': [0.18]
        })
        
        # Test preprocessing
        processed = processor.process_pipeline(test_data, normalize=False)
        print(f"✅ Data processor working")
        print(f"   Columns: {len(processed.columns)}")
        
        return True
    
    except Exception as e:
        print(f"❌ Data processor error: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Tax Evasion Detection System - Setup Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Project Structure", test_project_structure()))
    results.append(("Configuration", test_config()))
    results.append(("Sample Data", test_sample_data()))
    results.append(("Data Processor", test_data_processor()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All tests passed! Your setup is ready.")
        print("\n🚀 To start the application, run:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure API key in .env file")
        print("3. Ensure all directories and files exist")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
