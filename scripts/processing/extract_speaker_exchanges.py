"""
Extract shorter arguments from dialogues by pairing consecutive speaker turns.
"""

import json
import re
import os
import argparse
from datetime import datetime

def extract_speaker_turns(text):
    """Parse text into a list of {'speaker': 'speakerX', 'text': 'content'} dictionaries."""
    # Pattern matches "speaker1:", "speaker2:", etc.
    speaker_pattern = r'(speaker\d+):\s*'
    parts = re.split(speaker_pattern, text)
    
    turns = []
    # ['', 'speaker1', 'text1', 'speaker2', 'text2', ...]
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            speaker = parts[i]
            content = parts[i + 1].strip()
            turns.append({
                'speaker': speaker,
                'text': content
            })
    return turns

def extract_speaker_exchanges(input_file, output_file):
    """
    Extract speaker exchanges from dialogues.
    Each exchange consists of consecutive speaker turns (e.g., speaker1 -> speaker2).
    """
    exchanges = []
    dialogue_count = 0
    total_turns_mismatch = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            dialogue_count += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {dialogue_count}")
                continue

            input_text = data.get('input', '')
            output_text = data.get('output', '')
            
            input_turns = extract_speaker_turns(input_text)
            output_turns = extract_speaker_turns(output_text) if output_text else []
            
            # Check if output structure matches input structure (same number of turns)
            has_matching_output = (len(output_turns) == len(input_turns))
            if output_text and not has_matching_output:
                total_turns_mismatch += 1
            
            # Create exchanges from consecutive speaker turns
            for i in range(len(input_turns) - 1):
                # Construct input string for the exchange
                exchange_input = f"{input_turns[i]['speaker']}: {input_turns[i]['text']} {input_turns[i+1]['speaker']}: {input_turns[i+1]['text']}"
                
                exchange = {
                    'dialogue_id': dialogue_count,
                    'exchange_id': i + 1,
                    'total_exchanges': len(input_turns) - 1,
                    'input': exchange_input
                }
                
                # If output structure matches, we split the output too
                if has_matching_output:
                    exchange_output = f"{output_turns[i]['speaker']}: {output_turns[i]['text']} {output_turns[i+1]['speaker']}: {output_turns[i+1]['text']}"
                    exchange['output'] = exchange_output
                elif output_text:
                    pass 

                exchanges.append(exchange)
    
    if total_turns_mismatch > 0:
        print(f"Warning: {total_turns_mismatch} dialogues had mismatched input/output turn counts and their outputs were not split.")

    # Write exchanges to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for exchange in exchanges:
            f.write(json.dumps(exchange, ensure_ascii=False) + '\n')
    
    return len(exchanges), dialogue_count

def main():
    parser = argparse.ArgumentParser(description="Split dialogues into speaker exchanges.")
    parser.add_argument("--input_file", "-i", type=str, help="Path to input JSONL file")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Directory to save output")
    args = parser.parse_args()

    # Default paths if not provided
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if args.input_file:
        input_file = args.input_file
    else:
        default_input = os.path.join(base_dir, 'data', 'dialogue', 'out_dial_jsonl', 'dev_labeled.jsonl')
        if os.path.exists(default_input):
            input_file = default_input
        else:
             input_file = os.path.join(base_dir, 'data', 'dialogue', 'out_dial_jsonl', 'dev.jsonl')

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(base_dir, 'data', 'processed')
    
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.basename(input_file)
    name_part = os.path.splitext(input_filename)[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'speaker_exchanges_{name_part}_{timestamp}.jsonl'
    output_file = os.path.join(output_dir, output_filename)
    
    print(f"Extracting speaker exchanges from {input_file}...")
    try:
        num_exchanges, num_dialogues = extract_speaker_exchanges(input_file, output_file)
        
        print(f"\n✓ Extraction complete!")
        print(f"  - Processed {num_dialogues} dialogues")
        print(f"  - Extracted {num_exchanges} speaker exchanges")
        print(f"  - Output saved to: {output_file}")
        
        if num_exchanges > 0:
            print(f"\n--- Sample Exchange ---")
            with open(output_file, 'r', encoding='utf-8') as f:
                sample = json.loads(f.readline())
                print(f"Dialogue {sample.get('dialogue_id')}, Exchange {sample.get('exchange_id')}/{sample.get('total_exchanges')}")
                print(f"\nInput preview (first 300 chars):")
                print(sample.get('input', '')[:300] + "...")
                if 'output' in sample:
                    print(f"\nOutput preview (first 300 chars):")
                    print(sample.get('output', '')[:300] + "...")
    except FileNotFoundError:
        print(f"Error: Input file matches nothing: {input_file}")

if __name__ == '__main__':
    main()
