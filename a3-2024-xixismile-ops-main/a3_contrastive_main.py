#!/usr/bin/env python

import os
import json 
import torch
import logging
import argparse
import numpy as np

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from tqdm import tqdm
from datasets import load_dataset

from a3_contrastive_decoding import greedy_search_with_contrastive_decoding, greedy_search

################################################################################
# NOTE: Caution - only modify the TODO part!
################################################################################

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def seed_everything(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def out_file(outfile_path, generation_lst):
    with open(outfile_path, 'w') as fout:
        json.dump(generation_lst, fout)

    print(f'Saved to {outfile_path}')

def format_out(generated_text, prompt, gold_ref=None):
    output = {
        'prompt'  : prompt,
        'gen_text': generated_text, 
        'gold_ref': gold_ref, 
    } 
    return output 
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--amateur_name_or_path",
        default=None,
        type=str,
        help="Path to pre-trained model or shortcut name",
    )

    parser.add_argument(
        "--decoding-mode",
        default="greedy",
        type=str,
        help="Decoding mode",
        choices=["greedy", "contrastive"],
    )

    parser.add_argument("--outfile", type=str, default="part2_outfile.json")
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=50)

    parser.add_argument("--amateur_temperature", type=float, default=1.0)    
    parser.add_argument("--alpha", type=float, default=0.1)    
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args 

def main(args):
    logger.warning(f"device: {args.device}")

    if args.decoding_mode == "contrastive":
        print("> Using Contrastive Decoding")
    elif args.decoding_mode == "greedy":
        print("> Using Greedy Decoding")
    else:
        raise ValueError(f"Unsupported Decoding Method: {args.decoding_mode}")
    
    seed_everything(args)

    # Load GPT2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    # Set the pad token to be the eos token since GPT2 does not have a pad token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the expert model
    expert_model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path).to(args.device)

    if args.decoding_mode == "contrastive":
        assert args.amateur_name_or_path is not None
        # Load the amateur model
        amateur_model = GPT2LMHeadModel.from_pretrained(args.amateur_name_or_path).to(args.device)
   
    logger.info(args)

    # Load subset of the CC-News dataset
    dataset = load_dataset('json', data_files=os.path.join("data", "ccnews_100.json"))["train"] 
    text_column_name = "text" 

    def tokenize_function(examples):
        examples[text_column_name] = [tokenizer.bos_token + " " + x for x in examples[text_column_name] if len(x) > 0]

        result_dict = tokenizer(examples[text_column_name], add_special_tokens=False) 

        # NOTE: uses the first 32 tokens as the prompt, and the whole sentence as reference
        input_ids_lst = [x[:32] for x in result_dict['input_ids'] if len(x) >= 160 ]
        gold_lst = [x for x in result_dict['input_ids'] if len(x) >= 160 ]

        return {'input_ids': input_ids_lst, 'gold': gold_lst}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
    )

    print(tokenized_dataset)

    prompt_ids = tokenized_dataset['input_ids'] 
    ref_lst = tokenized_dataset['gold'] 
    ref_lst = tokenizer.batch_decode(ref_lst)

    prompt_lst = tokenizer.batch_decode(prompt_ids)
    print(len(prompt_lst), prompt_lst[:20])

    generation_lst = []
    
    prompt_lst = prompt_lst[:args.num_generations]
    for iidx, prompt_text in tqdm(enumerate(prompt_lst), total=len(prompt_lst)):

        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(args.device)

        if args.decoding_mode == "contrastive":
            output_sequences = greedy_search_with_contrastive_decoding(
                model=expert_model,
                amateur_model=amateur_model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                alpha=args.alpha,
                amateur_temperature=args.amateur_temperature,
            )

        elif args.decoding_mode == "greedy":
            output_sequences = greedy_search(
                model=expert_model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            raise ValueError(f"Unsupported Decoding Method: {args.decoding_mode}")
        
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequence = output_sequences[0].tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        
        generated_dict = format_out(text, prompt_text, gold_ref=ref_lst[iidx])
        generation_lst.append(generated_dict)
        print(text)

    out_file(args.outfile, generation_lst)
    return generation_lst
    
    
if __name__ == "__main__":
    args = get_args() 
    main(args)