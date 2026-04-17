import json
import argparse
import re
from collections import defaultdict
import os

def extract_speaker_turns(text):
    """
    Parse text into a list of {'speaker': 'speakerX', 'text': 'content'} dictionaries.
    """
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

def merge_turn(existing, new):
    """
    Merge two versions of a turn. 
    """
    if existing is None:
        return new
    
    # Count annotation tags
    existing_tags = existing['text'].count("IMPLICIT_")
    new_tags = new['text'].count("IMPLICIT_")
    
    if new_tags > existing_tags:
        return new
    elif new_tags < existing_tags:
        return existing
    else:
        if len(new['text']) > len(existing['text']):
            return new
        return existing

def process_files(reconstructed_file, original_file, output_file):
    print(f"Reading original file: {original_file}")
    original_data = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    original_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in original file: {line[:50]}...")
                
    print(f"Reading reconstructed file: {reconstructed_file}")
    reconstructed_data = []
    with open(reconstructed_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    reconstructed_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON in reconstructed file: {line[:50]}...")
    
    # Check alignment
    if len(original_data) != len(reconstructed_data):
        print(f"Warning: Files have different line counts ({len(original_data)} vs {len(reconstructed_data)}).")
    
    # Group by dialogue_id
    dialogues = defaultdict(list)
    
    for i, rec_entry in enumerate(reconstructed_data):
        # Use index from file if available, else use list index
        idx = rec_entry.get('index', i)
        
        if idx < len(original_data):
            orig_entry = original_data[idx]
            
            # Helper to get ID safely
            dialogue_id = orig_entry.get('dialogue_id')
            exchange_id = orig_entry.get('exchange_id')
            
            if dialogue_id is not None and exchange_id is not None:
                dialogues[dialogue_id].append({
                    'exchange_id': exchange_id,
                    'reconstructed_text': rec_entry.get('reconstructed_text', '')
                })
        else:
            print(f"Skipping index {idx} - out of bounds for original file")

    results = []
    
    print(f"Processing {len(dialogues)} dialogues...")
    
    for dialogue_id, exchanges in dialogues.items():
        # Sort exchanges by ID
        exchanges.sort(key=lambda x: x['exchange_id'])
        
        
        if not exchanges:
            continue
            
        max_eid = exchanges[-1]['exchange_id']
        num_turns = max_eid + 1
        
        turns = [None] * num_turns
        
        for ex in exchanges:
            eid = ex['exchange_id'] # 1-based
            parts = extract_speaker_turns(ex['reconstructed_text'])
            
            t1_idx = eid - 1
            t2_idx = eid
            
            if len(parts) >= 1:
                if t1_idx < num_turns:
                    turns[t1_idx] = merge_turn(turns[t1_idx], parts[0])
            
            if len(parts) >= 2:
                if t2_idx < num_turns:
                    turns[t2_idx] = merge_turn(turns[t2_idx], parts[1])
            
                 
        # Join turns into full dialogue
        full_text_parts = []
        valid = True
        
        # Check for missing turns (gaps in exchanges)
        for i in range(num_turns):
            if turns[i] is None:
                pass 
            else:
                full_text_parts.append(f"{turns[i]['speaker']}: {turns[i]['text']}")
            
        full_text = " ".join(full_text_parts)
            
        if full_text:
            results.append({
                'dialogue_id': dialogue_id,
                'reconstructed_text': full_text
            })
            
    # Write output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Successfully joined dialogues. Output written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join reconstructed speaker exchanges back into full dialogues.")
    parser.add_argument("--reconstructed_file", type=str, required=True, help="Path to the reconstructed JSONL file (output of generation)")
    parser.add_argument("--original_file", type=str, required=True, help="Path to the original speaker exchanges JSONL file (input to generation)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the joined dialogues JSONL")
    args = parser.parse_args()
    
    process_files(args.reconstructed_file, args.original_file, args.output_file)
