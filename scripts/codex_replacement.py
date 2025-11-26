#!/usr/bin/env python3
"""
Codex CLI Replacement - Drop-in replacement for OpenAI Codex CLI
Uses OpenAI API directly to avoid Codex CLI hanging issues
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI

def main():
    parser = argparse.ArgumentParser(
        description="Codex CLI replacement using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt", nargs="*", help="Command or prompt")
    parser.add_argument("--model", default="gpt-4", help="Model to use (default: gpt-4)")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        print("   Set it with: export OPENAI_API_KEY='your-key'", file=sys.stderr)
        sys.exit(1)
    
    # Get prompt
    if args.prompt:
        prompt = " ".join(args.prompt)
    else:
        # Read from stdin if no prompt provided
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Usage: codex <prompt>", file=sys.stderr)
            sys.exit(1)
    
    # Initialize client
    try:
        client = OpenAI(api_key=api_key, timeout=args.timeout)
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Make request
    try:
        print(f"🤖 Using model: {args.model}", file=sys.stderr)
        print(f"📝 Processing: {prompt[:50]}{'...' if len(prompt) > 50 else ''}", file=sys.stderr)
        print("", file=sys.stderr)
        
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=args.timeout
        )
        
        # Print response
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()











