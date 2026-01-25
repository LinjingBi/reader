#!/usr/bin/env python3
"""List available Gemini models using the google-genai SDK"""

import os
import sys
from google import genai

def list_models(api_key: str = None):
    """List all available Gemini models"""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("Error: No API key provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        print("Or pass it as an argument: python list_gemini_models.py YOUR_API_KEY")
        sys.exit(1)
    
    try:
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        
        print("Available Gemini Models:")
        print("=" * 60)
        
        free_tier_models = []
        paid_models = []
        
        for model in models:
            model_name = model.name if hasattr(model, 'name') else str(model)
            # Common free tier models
            if any(x in model_name.lower() for x in ['flash', 'gemini-1.5-flash', 'gemini-2.0-flash']):
                free_tier_models.append(model_name)
            else:
                paid_models.append(model_name)
        
        if free_tier_models:
            # print("\n🆓 Free Tier Models (typically):")
            for model in sorted(free_tier_models):
                print(f"  • {model}")
        
        if paid_models:
            # print("\n💰 Paid/Pro Models:")
            for model in sorted(paid_models):
                print(f"  • {model}")
        
        print("\n" + "=" * 60)
        print(f"Total models found: {len(free_tier_models) + len(paid_models)}")
        print("\nNote: Free tier availability may vary. Check https://ai.google.dev/pricing for current details.")
        
    except Exception as e:
        print(f"Error listing models: {e}")
        print("\nMake sure you have google-genai installed:")
        print("  pip install google-genai")
        sys.exit(1)

if __name__ == "__main__":
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    list_models(api_key)

