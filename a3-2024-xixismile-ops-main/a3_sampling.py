import torch
from typing import Any, Dict
from a3_utils import *


class TopKSamplerForCausalLM(GeneratorForCausalLM):
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
    def sample(
        self,
        inputs: dict,
        top_k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-k sampling. 
        This means that we sample the next token from the top-k scoring tokens 
        by using their probability values.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - It only handles top_k => 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_k (int): the number of highest probability vocabulary tokens 
                         to keep for top-k filtering/sampling
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-k sampled sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens, top_k=top_k)
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

        updated_inputs = inputs.copy()
        for i in range(max_new_tokens):

            model_out = self.model(**updated_inputs)
            logits = model_out.logits

            temp_adjusted_probs = torch.nn.functional.softmax(logits[:, -1, :] / temperature, dim=-1)

            top_probs, top_indices = torch.topk(temp_adjusted_probs, top_k, dim=-1)
            norm_probs = top_probs / torch.sum(top_probs, dim=-1, keepdim=True)

            chosen_token_idx = torch.multinomial(norm_probs, 1).item()
            chosen_token_id = top_indices[0][chosen_token_idx].unsqueeze(0)

            updated_inputs = self.prepare_next_inputs(updated_inputs, chosen_token_id.unsqueeze(0), use_cache=False,
                                                          use_cuda=True)

            if updated_inputs["input_ids"][0][-1] == self.tokenizer.eos_token_id:
                break
        gen_seq = updated_inputs["input_ids"]
        return gen_seq

class TopPSamplerForCausalLM(GeneratorForCausalLM):
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
    def sample(
        self,
        inputs: dict,
        top_p: float,
        temperature: float,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-p sampling. 
        This means that we sample the next token from the smallest set of most 
        probable tokens with probabilities that cumulatively add up to top_p *or higher*.
        If there are no tokens falling in the top_p cumulative probability mass 
        (e.g. because the top scoring tokens probability is larger than top_p) 
        then samples the top scoring token.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_p (float): the cumulative probability mass to select the smallest 
                           set of most probable tokens with probabilities that 
                           cumulatively add up to top_p or higher.
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-p sampled sequence made of token ids of size (1,generated_seq_len)
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
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################
        updated_inputs = inputs.copy()
        for i in range(max_new_tokens):

            outputs = self.model(**updated_inputs )
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits[:, -1, :] / temperature, dim=-1) # (1, vocab_size)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
           
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).squeeze(0)
            candidate_indices = sorted_indices[0][cumulative_probs <= top_p] 

            if len(candidate_indices) == 0:
                next_token_id = sorted_indices[0][0].unsqueeze(0)
            else:
                next_token_index = torch.multinomial(probs[0][candidate_indices], 1).item()
                next_token_id = candidate_indices[next_token_index].unsqueeze(0)

            updated_inputs = self.prepare_next_inputs(updated_inputs, next_token_id.unsqueeze(0), use_cache = False, use_cuda = True)
          
            if updated_inputs ["input_ids"][0][-1] == self.tokenizer.eos_token_id:
                break

        generated_seq = updated_inputs["input_ids"]
        return generated_seq

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