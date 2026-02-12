"""
Quick test script to verify OpenAI API key validity
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

# Load environment variables
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('OPENAI_MODEL', 'gpt-4o')

print(f"Testing OpenAI API connection...")
print(f"API Key starts with: {api_key[:15] if api_key else 'NOT FOUND'}...")
print(f"API Key length: {len(api_key) if api_key else 0} characters")
print(f"Model: {model}")
print("-" * 50)

try:
    # Create client with a timeout
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(30.0, connect=10.0)  # 30s total, 10s connect
    )
    
    print("Making API call...")
    
    # Make a minimal test call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Return only the JSON: {\"status\": \"ok\"}"}
        ],
        temperature=0.0,
        max_tokens=20,
        response_format={"type": "json_object"}
    )
    
    result = response.choices[0].message.content
    print("✅ SUCCESS!")
    print(f"Response: {result}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print(f"Model used: {response.model}")
    
except Exception as e:
    print("❌ ERROR!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    
    # Check for common issues
    if "api_key" in str(e).lower() or "auth" in str(e).lower():
        print("\n🔑 API KEY ISSUE DETECTED!")
        print("Your API key may be:")
        print("  - Invalid or expired")
        print("  - Not authorized for the model")
        print("  - Missing required permissions")
        print("\nPlease verify your API key at: https://platform.openai.com/api-keys")
    elif "timeout" in str(e).lower():
        print("\n⏱️ TIMEOUT ISSUE DETECTED!")
        print("Network connection may be slow or blocked")
    elif "rate" in str(e).lower() or "quota" in str(e).lower():
        print("\n📊 RATE LIMIT/QUOTA ISSUE!")
        print("Check your usage at: https://platform.openai.com/usage")
