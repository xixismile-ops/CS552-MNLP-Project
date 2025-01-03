import mauve 
import json
import argparse

from transformers import AutoTokenizer 

################################################################################
# NOTE: Caution - do not modify this file!! 
#       If you do, it won't affect your grading, but you might not get 
#       the right results when testing with the notebook.
################################################################################

def load_file(filepath):
    text_list = []
    text_refs = []
    with open(filepath, "r") as fin:
        examples = json.load(fin)

    for example in examples:
        text_pre = example['prompt']
        text_gen = example['gen_text'].replace(text_pre, "").strip()
        text_ref = example['gold_ref'].replace(text_pre, "").strip()

        text_list.append(text_gen)
        text_refs.append(text_ref)
    return text_list, text_refs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="outfile.json",
        type=str,
        help="Path to output file",
    )

    args = parser.parse_args()

    cumulative_stats = {}
    text_gens, text_gold = load_file(args.filepath)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print(len(text_gens), len(text_gold))
    tgt_len = 128

    x = tokenizer(text_gens, truncation=True, max_length=tgt_len)['input_ids']
    y = tokenizer(text_gold, truncation=True, max_length=tgt_len)['input_ids']
  
    xxyy = [(xx, yy) for (xx, yy) in zip(x, y) if len(xx) == tgt_len and len(yy) == tgt_len]
    x, y = zip(*xxyy)
    
    text_gens = tokenizer.batch_decode(x)
    text_gold = tokenizer.batch_decode(y)
    print(len(text_gens), len(text_gold))
    
    out = mauve.compute_mauve(p_text=text_gens, q_text=text_gold, device_id=0, max_text_length=256, 
        verbose=False, featurize_model_name='gpt2')

    print(f"Mauve Score: {out.mauve}") 
    cumulative_stats['mauve'] = out.mauve    