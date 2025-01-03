import torch
from typing import Any, Dict
from a3_utils import *


class GreedySearchDecoderForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    
    @torch.no_grad()
    def search(
        self,
        inputs: dict,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using greedy decoding. 
        This means that we always pick the next token with the highest score/probability.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping (i.e. if the next token is an EOS 
            (end-of-sentence) token, you should stop decoding) or stops at 
            max_new_tokens.
        - It only handles inputs of batch size = 1.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: greedy decoded best sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # Hint (#1): There are 2 ways to pass the inputs to the model. Please open the
        # [GPT2LMHeadModel documentation](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel) 
        # and read the `Parameters` and `Returns` sections while looking at these hints.
        # Either of these approaches is fine since we don't expect the most efficient solution:
        #
        #   1. **If recomputing the past decoder processes at each decoding step:**
        #       Since the tokenizer's output dictionary keys matches these `Parameter` 
        #       i.e. arguments of the model you can directly do:
        #       ```python
        #       >> self.model(**inputs)
        #       ```
        #       Just be careful and think about how you modify the "input_ids" 
        #       and "attention_mask" keys across decoding steps. 
        
        #   2. **If using cached decoder hidden states at each decoding step:**
        #       To speed up the process (although *not required*) you can also get 
        #       the computed key/values hidden-states *so far* with `use_cache=True`
        #       where in the first step you may need to do:
        #       ```python
        #       >> self.model(**inputs, use_cache=True)
        #       ```
        #       This will return an extra dictionary entry called "past_key_values".
        #       In the next steps you would do, assuming your previous output 
        #       dict is called `outputs`:
        #       ```python
        #       >> self.model(**inputs, use_cache=True, past_key_values=outputs["past_key_values"])
        #       ```
        #       Again, be careful as to how you modify the "input_ids" and 
        #       "attention_mask" keys across decoding steps. In particular the 
        #       cached setting expects them to be different from the non-cached 
        #       setting. Read the `Parameters` and `Returns` sections of the 
        #       GPT2LMHeadModel carefully.
        #
        # Hint (#2): You can implement and use the `self.prepare_next_inputs` 
        #   function in `a3_utils.py` inherited by all decoding and sampling classes 
        #   (although you are not required to) to reduce repeated code and make it
        #   more readable. There isn't a unique solution for this so use it as you wish
        #   or create another function in this super class.
        ########################################################################

        input_state = inputs.copy()


        for _ in range(max_new_tokens):

            model_prediction = self.model(**input_state)
            prediction_logits = model_prediction.logits
            highest_prob_token = torch.argmax(prediction_logits[:, -1, :], dim=-1).unsqueeze(0)

            input_state = self.update_input_state(input_state, highest_prob_token, cache_enabled=False,
                                                  cuda_enabled=True)

            if input_state["input_ids"][0, -1] == self.tokenizer.eos_token_id:
                break

        # 返回包含所有生成token的序列
        generated_sequence = input_state["input_ids"]
        return generated_sequence

class BeamSearchDecoderForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    
    @torch.no_grad()
    def search(
        self,
        inputs: dict,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences=1,
        length_penalty: float = 0.0
    ) -> dict: 
        """Generates sequences of token ids with self.model, 
        (which has a language modeling head) using beam search. This means that 
        given a probability distribution over the possible next tokens and 
        a beam width (here num_beams), needs to keep track of the most probable 
        num_beams candidates. (Hint: use log probabilities!)

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens. 
        - It only handles inputs of batch size = 1.
        - It only handles beam size > 1.
        - It includes a length_penalty variable that controls the score assigned 
            to a long generation. This is implemented by exponiating the amount 
            of newly generated tokens to this value. Then, divide the score which 
            can be calculated as the sum of the log probabilities so far.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)
            num_beams (int): number of beams for beam search
            num_return_sequences (int, optional):
                the amount of best sequences to return. Cannot be more than beam size.
                Defaults to 1.
            length_penalty (float, optional): 
                exponential penalty to the length that is used with beam-based generation. 
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. 
                Defaults to 0.0.

        Returns:
            dict: dictionary with two key values:
                    - "sequences": torch.LongTensor depicting the best generated sequences (token ID tensor) 
                        * shape (num_return_sequences, maximum_generated_sequence_length)
                        * ordered from best scoring sequence to worst
                        * if a sequence has reached end of the sentence, 
                          you can fill the rest of the tensor row with the pad token ID
                    - "sequences_scores": length penalized log probability score list, ordered by best score to worst
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(
            inputs, 
            max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################
        
        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################

        device = inputs["input_ids"].device

        def get_candidates(log_probs, count):
            topk_scores, topk_indices = torch.topk(log_probs, count)
            return [topk_indices[i].unsqueeze(0) for i in range(count)], topk_scores

        def prepare_next_inputs_for_beams(inputs, new_token_ids, use_cuda):
            return [self.prepare_next_inputs(inputs, token_id.unsqueeze(0), use_cache=False, use_cuda=use_cuda) for
                    token_id in new_token_ids]

        def update_beams(beam_inputs, scores, top_candidates, penalty_factor):
            selected_inputs = [beam_inputs[i] for i in top_candidates.indices]
            if penalty_factor != 0:
                adjusted_scores = torch.index_select(scores, 0, top_candidates.indices)
            else:
                adjusted_scores = top_candidates.values
            return selected_inputs, adjusted_scores

        current_input = inputs.copy()
        beam_inputs, beams_scores = [], torch.tensor([]).to(device)

        for step in range(max_new_tokens):
            if step == 0:
                model_outputs = self.model(**current_input)
                log_probs = torch.nn.functional.log_softmax(model_outputs.logits, dim=-1).squeeze(0)[-1]
                new_token_ids, beam_scores = get_candidates(log_probs, num_beams)
                beam_inputs = prepare_next_inputs_for_beams(current_input, new_token_ids, use_cuda=True)
            else:
                updated_beam_inputs, updated_beam_scores = [], torch.tensor([], device=device)
                for beam_index, beam_input in enumerate(beam_inputs):
                    if beam_input["input_ids"][0, -1] == self.tokenizer.eos_token_id:
                        updated_beam_inputs.append(beam_input)
                        updated_beam_scores = torch.cat((updated_beam_scores, beams_scores[beam_index].unsqueeze(0)))
                        continue

                    model_outputs = self.model(**beam_input)
                    log_probs = torch.nn.functional.log_softmax(model_outputs.logits, dim=-1).squeeze(0)[-1]
                    new_token_ids, new_scores = get_candidates(log_probs, num_beams)

                    for idx, token_id in enumerate(new_token_ids):
                        next_input = self.prepare_next_inputs(beam_input, token_id.unsqueeze(0), use_cache=False,
                                                              use_cuda=True)
                        updated_beam_inputs.append(next_input)
                        score = (new_scores[idx] + beams_scores[beam_index]).unsqueeze(0)
                        updated_beam_scores = torch.cat((updated_beam_scores, score), dim=0)

                top_candidates = torch.topk(updated_beam_scores, num_beams)
                beam_inputs, beams_scores = update_beams(updated_beam_inputs, updated_beam_scores, top_candidates,
                                                         length_penalty)

        # 生成最终序列和得分
        final_sequences = [beam_input["input_ids"][0].tolist() for beam_input in beam_inputs[:num_return_sequences]]
        final_scores = beams_scores[:num_return_sequences] if length_penalty == 0 else torch.topk(beams_scores, num_return_sequences).values

        generated = {"sequences": torch.tensor(final_sequences), "sequences_scores": final_scores}
        return generated

def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "vicgalle/gpt2-alpaca-gpt4"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == '__main__':
    main()