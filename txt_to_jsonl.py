#!/usr/bin/env python3
"""
Convert readable TXT format back to JSONL annotation files.

This script reads TXT files (created by jsonl_to_readable.py) and converts
them back to the original JSONL format. This is useful after manually
correcting the annotations in the readable text format.
"""

import json
import re
from pathlib import Path


def parse_txt_to_jsonl(input_txt_path, output_jsonl_path):
    """
    Convert TXT annotation file back to JSONL format.

    Args:
        input_txt_path: Path to input TXT file
        output_jsonl_path: Path to output JSONL file
    """
    sentences = []
    current_sentence = None
    current_tokens = []
    current_text = None

    with open(input_txt_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check for sentence header
        if line.startswith('='*80):
            # If we have a previous sentence, save it
            if current_sentence is not None and current_text is not None:
                sentences.append({
                    'text': current_text,
                    'annotations': {
                        'tokens': current_tokens
                    }
                })

            # Reset for new sentence
            current_tokens = []
            current_text = None
            current_sentence = None

            # Skip the separator line
            i += 1
            if i < len(lines):
                # Read sentence number line
                sentence_line = lines[i].strip()
                if sentence_line.startswith('SENTENCE '):
                    current_sentence = int(sentence_line.split()[1])
                i += 1

            # Skip another separator line
            if i < len(lines) and lines[i].strip().startswith('='*80):
                i += 1

            # Read the TEXT line
            if i < len(lines) and lines[i].strip().startswith('TEXT: '):
                current_text = lines[i].strip()[6:]  # Remove 'TEXT: ' prefix
                i += 1

            # Skip empty line
            if i < len(lines) and not lines[i].strip():
                i += 1

            # Skip header line (IDX TOKEN POS...)
            if i < len(lines) and lines[i].strip().startswith('IDX'):
                i += 1

            # Skip separator line (---- ----)
            if i < len(lines) and lines[i].strip().startswith('-'):
                i += 1

            continue

        # Check if this is a token line
        # Token lines start with a number (the index)
        if line and not line.startswith('='*80):
            # Split by whitespace, but preserve the structure
            parts = line.split(None, 6)  # Split into max 7 parts

            if len(parts) >= 7:
                try:
                    token_idx = int(parts[0])
                    token_text = parts[1]
                    pos = parts[2]
                    tag = parts[3]
                    lemma = parts[4]
                    dep = parts[5]
                    ent = parts[6]

                    token = {
                        'text': token_text,
                        'pos': pos,
                        'tag': tag,
                        'lemma': lemma,
                        'dep': dep,
                        'ent': ent
                    }
                    current_tokens.append(token)
                except (ValueError, IndexError):
                    # Not a valid token line, skip it
                    pass

        i += 1

    # Don't forget the last sentence
    if current_sentence is not None and current_text is not None:
        sentences.append({
            'text': current_text,
            'annotations': {
                'tokens': current_tokens
            }
        })

    # Write JSONL file
    with open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for sentence in sentences:
            outfile.write(json.dumps(sentence, ensure_ascii=False) + '\n')

    print(f"✓ Converted {input_txt_path.name} to {output_jsonl_path.name}")
    print(f"  - {len(sentences)} sentences processed")


def validate_jsonl_structure(jsonl_path):
    """
    Validate the structure of a JSONL file.

    Args:
        jsonl_path: Path to JSONL file to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                try:
                    data = json.loads(line.strip())

                    # Check required keys
                    if 'text' not in data:
                        errors.append(f"Line {line_num}: Missing 'text' field")

                    if 'annotations' not in data:
                        errors.append(f"Line {line_num}: Missing 'annotations' field")
                    elif 'tokens' not in data['annotations']:
                        errors.append(f"Line {line_num}: Missing 'tokens' field in annotations")
                    else:
                        # Check each token
                        for tok_idx, token in enumerate(data['annotations']['tokens']):
                            required_fields = ['text', 'pos', 'tag', 'lemma', 'dep', 'ent']
                            for field in required_fields:
                                if field not in token:
                                    errors.append(
                                        f"Line {line_num}, Token {tok_idx}: Missing '{field}' field"
                                    )

                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")

    except Exception as e:
        errors.append(f"Error reading file: {e}")

    return len(errors) == 0, errors


def main():
    """Main function to convert all TXT files back to JSONL."""

    # Define input and output directories
    input_txt_dir = Path("validated_test_data_readable/txt")
    output_jsonl_dir = Path("validated_test_data/validated_jsonl")

    # Create output directory
    output_jsonl_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_jsonl_dir}")
    print()

    # Get all TXT files
    txt_files = list(input_txt_dir.glob("*.txt"))

    if not txt_files:
        print(f"No TXT files found in {input_txt_dir}")
        print("Please ensure TXT files exist in the validated_test_data_readable/txt directory.")
        return

    print(f"Found {len(txt_files)} TXT files to convert\n")
    print(f"{'='*80}")

    # Convert each file
    all_valid = True
    for txt_file in sorted(txt_files):
        print(f"\nProcessing: {txt_file.name}")

        # Generate output filename
        base_name = txt_file.stem
        jsonl_file = output_jsonl_dir / f"{base_name}.jsonl"

        # Convert TXT to JSONL
        parse_txt_to_jsonl(txt_file, jsonl_file)

        # Validate the output
        print(f"Validating {jsonl_file.name}...")
        is_valid, errors = validate_jsonl_structure(jsonl_file)

        if is_valid:
            print(f"  ✓ Validation passed")
        else:
            print(f"  ✗ Validation failed with {len(errors)} errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more errors")
            all_valid = False

    print(f"\n{'='*80}")
    print(f"Conversion complete!")
    print(f"JSONL files saved to: {output_jsonl_dir}")

    if all_valid:
        print(f"✓ All conversions validated successfully!")
    else:
        print(f"⚠ Some conversions had validation errors. Please review.")


if __name__ == "__main__":
    main()
