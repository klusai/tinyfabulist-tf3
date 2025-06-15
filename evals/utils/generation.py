"""
Text generation utilities
"""

import torch
from typing import List, Dict, Any
import sys
import os

# Add lib directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.logging_utils import get_logger

logger = get_logger()


def generate_completions(model, tokenizer, test_data: List[Dict[str, Any]], 
                        max_new_tokens: int = 256, temperature: float = 0.8, 
                        device: str = "cpu") -> List[str]:
    """
    Generate text completions for a list of prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        test_data: List of test samples with 'prompt' key
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to run on
        
    Returns:
        List of generated completions
    """
    
    model.eval()
    completions = []
    
    logger.info(f"Generating {len(test_data)} completions...")
    
    for i, sample in enumerate(test_data):
        prompt = sample['prompt']
        
        try:
            completion = _generate_single_completion(
                model, tokenizer, prompt, max_new_tokens, temperature, device
            )
            completions.append(completion)
            
            if i < 3:  # Log first few samples
                logger.info(f"--- Sample {i+1} ---")
                logger.info(f"Prompt: {prompt[:100]}...")
                logger.info(f"Completion: {completion[:100]}...")
                logger.info("-" * 20)
                
        except Exception as e:
            logger.warning(f"Failed to generate completion for sample {i}: {e}")
            completions.append("")  # Empty completion on error
    
    logger.info("✓ Completions generated.")
    return completions


def _generate_single_completion(model, tokenizer, prompt: str, max_new_tokens: int, 
                               temperature: float, device: str) -> str:
    """Generate a single text completion"""
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode only the new tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return completion.strip() 