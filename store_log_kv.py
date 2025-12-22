'''Given a full reasoning trace, which KV vectors should we store so that they still encode the entire reasoning?'''

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache
import torch
from tiger_utils import read_json, write_pickle
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig
import re
import argparse

from block_attention import apply_pkv_rerotary_position_embeddings
from utils import STORE_PREFIX

# --- UTILS ---
def get_log_kv(text, tokenizer, model, emb):
  '''Convert text into KV cache, while removing positional bias'''
  input_ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(model.device)

  with torch.no_grad():
    outputs = model(
      input_ids=input_ids,
      past_key_values=DynamicCache(), # The concatenated KV values of logs are passed to LLM using past_key_values parameter
                                      # DynamicCache() → empty cache
      use_cache=True, # model returns KV for every layer
      output_attentions=False,
      output_hidden_states=False
    )

    # revert the KV value back
    kv = apply_pkv_rerotary_position_embeddings(pkv=outputs.past_key_values, emb=emb) # Remove positional embeddings
  
  kv.key_cache = [x.cpu() for x in kv.key_cache]
  kv.value_cache = [x.cpu() for x in kv.value_cache]
  return kv

# --- UTILS of the DEFAULT ENCODE-STORAGE STRATEGY ---
def get_reasoning_kv(log, tokenizer, model, emb, last_num: int):
  '''Encode the entire reasoning trace, but store only the last N responses'''
  assert (len(log) % 2 == 0) and len(log) >= 2 # Log format: [user, assistant, user, assistant, ...]

  # only use the assistant messages
  assistant_logs = [x for i, x in enumerate(log) if i % 2 == 1]
  assistant_logs_except_last = '\n\n'.join(assistant_logs[:-last_num]).replace('<|eot_id|>', '') # all assistant messages except the last N
  assistant_logs_last = '\n\n'.join(assistant_logs[-last_num:]).replace('<|eot_id|>', '') + '\n\n'# the last N assistant messages

  # encode (tokenize) the entire reasoning trace
  input_ids = tokenizer.encode(assistant_logs_except_last + '\n\n' + assistant_logs_last, return_tensors='pt', add_special_tokens=False) # all
  input_ids_except_last = tokenizer.encode(assistant_logs_except_last + '\n\n', return_tensors='pt', add_special_tokens=False) # all except the last N
  input_ids_last = tokenizer.encode(assistant_logs_last, return_tensors='pt', add_special_tokens=False) # the last N
  assert input_ids.shape[1] == input_ids_except_last.shape[1] + input_ids_last.shape[1] # verification

  kv = get_log_kv(assistant_logs_except_last + '\n\n' + assistant_logs_last, tokenizer, model, emb) # encode entire reasoning trace into KV
  assert kv.key_cache[0].shape[2] == input_ids.shape[1] # verification
  # kv.key_cache is a list of tensors, one per transformer layer, containing the Key vectors produced by self-attention for specific tokens.
  # kv.key_cache[layer].shape = (batch_size, num_heads, seq_len, head_dim)

  start_idx = input_ids_except_last.shape[1] # the start index (first token) of the last N responses in the full reasoning trace
  kv._seen_tokens = kv._seen_tokens - start_idx # update the number of seen tokens to only account for the last N responses (because of slicing below)
  num_layers = len(kv) 
  for layer_idx in range(num_layers):
    # for every layer, we throw away the KV values of all the tokens before the last N responses and only keep the tokens of the last N responses
    # slicing only return view, not changing the original tensor
    # pickle saves the underlying storage
    kv.key_cache[layer_idx] = kv.key_cache[layer_idx][:, :, start_idx:, :].clone().contiguous() 
    kv.value_cache[layer_idx] = kv.value_cache[layer_idx][:, :, start_idx:, :].clone().contiguous()
    assert kv.key_cache[layer_idx].shape == kv.value_cache[layer_idx].shape
    assert kv.key_cache[layer_idx].shape[2] == kv._seen_tokens == input_ids_last.shape[1]

  # NOW: we have a KV cache that contains only last-N tokens, but semantically encodes the full reasoning trace
  # not good to keep it on GPU, so move to CPU
  kv.key_cache = [x.cpu() for x in kv.key_cache]
  kv.value_cache = [x.cpu() for x in kv.value_cache]
  return kv

# --- DEFAULT ENCODE-STORAGE STRATEGY ---
def store_reasoning_offline(dataset, last_num: int):
  '''The default storage strategy: Encode all reasoning traces, store the KV values corresponding to the last_num reasoning trace
  (e.g., if last_num = 2, store the KV values corresponding to tokens in the last two reasoning traces)
  '''

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  logs = read_json(f'./data/{dataset}/preds/log.json')
  log_kvs = []

  for log in tqdm(logs):
    log = log[0]
    kv = get_reasoning_kv(log, tokenizer, model, emb, last_num)
    log_kvs.append(kv)

  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_{last_num}.pkl') # save the KV caches to disk as pickle files

# --- OTHER ENCODE-STORAGE STRATEGY ---
def store_reasoning_last_as_context(dataset):
  '''Encode the last reasoning trace, store the KV values corresponding to the last reasoning trace'''
  # Encodes only last assistant message
  # Stores its KV
  # No access to previous reasoning during encoding
  # This matches the KV-cache baseline in the paper.

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  logs = read_json(f'./data/{dataset}/preds/log.json') 
  log_kvs = []

  for log in tqdm(logs):
    log = log[0]
    assert (len(log) % 2 == 0) and len(log) >= 2
    assistant_logs_last = log[-1].replace('<|eot_id|>', '') + '\n\n'
    kv = get_log_kv(assistant_logs_last, tokenizer, model, emb)
    log_kvs.append(kv)

  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_as_context.pkl') # save the KV caches to disk as pickle files

# --- OTHER ENCODE-STORAGE STRATEGY ---
def store_reasoning_all(dataset):
  '''Encode all reasoning traces, store the KV values corresponding to all reasoning traces'''
  # Encodes all assistant messages
  # Stores everything
  # Massive storage
  # No compression

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  logs = read_json(f'./data/{dataset}/preds/log.json')
  log_kvs = []

  for log in tqdm(logs):
    log = log[0]
    assert (len(log) % 2 == 0) and len(log) >= 2
    assistant_logs = [x for i, x in enumerate(log) if i % 2 == 1]
    assistant_logs = '\n\n'.join(assistant_logs).replace('<|eot_id|>', '') + '\n\n'
    kv = get_log_kv(assistant_logs, tokenizer, model, emb)
    log_kvs.append(kv)

  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/kv/reasoning_all.pkl') # save the KV caches to disk as pickle files


# --- UTILS ---
def extract_action(text: str):
  '''extract the last action from the text'''
  # This function defines what “last action” means.
  # < > </ > tokens always seem to be independent in the token space
  # if <ans> or <keywords> exist, then use them; otherwise, use the entire trace

  # search for an answer
  answer = None
  if '<ans>' in text and '</ans>' in text:
    _answer = re.findall(r'<ans>(.*?)</ans>', text)
    _answer = [x for x in _answer if x.strip() != '']
    if len(_answer) != 0:
      answer = f'<ans>{_answer[-1]}</ans>'
  
  if answer is not None:
    return answer
  
  # search for a keyword
  keywords = None
  if '<keywords>' in text and '</keywords>' in text:
    _keywords = re.findall(r'<keywords>(.*?)</keywords>', text)
    _keywords = [x for x in _keywords if x.strip() != '']
    if len(_keywords) != 0:
      keywords = f'<keywords>{_keywords[-1]}</keywords>'
  
  if keywords is not None:
    return keywords
  
  # TODO: this is only extracting the last subquestion, (maybe merge all subquestions?)
  question = None
  if '<subquestion>' in text and '</subquestion>' in text:
    _question = re.findall(r'<subquestion>(.*?)</subquestion>', text)
    _question = [x for x in _question if x.strip() != '']
    if len(_question) != 0:
      question = f'<subquestion>{_question[-1]}</subquestion>'
  
  if question is not None:
    return question

  return text

# --- UTILS ---
def sublist_indices(lst, sublst):
  sub_len = len(sublst)
  for i in range(len(lst) - sub_len + 1):
    if torch.equal(lst[i:i+sub_len], sublst):
      return i, i + sub_len  # start index, end index
  assert False

# --- OTHER ENCODE-STORAGE STRATEGY ---
def store_reasoning_last_action(dataset: str):
  '''Encode all reasoning traces, store the KV values corresponding to the last agentic action in the reasoning trace'''

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  logs = read_json(f'./data/{dataset}/preds/log.json')
  log_kvs = []

  for log in tqdm(logs):
    # use only assistant messages
    log = [x for i, x in enumerate(log[0]) if i % 2 == 1]
    action = extract_action(log[-1]).replace('<|eot_id|>', '')
    
    # rsplit so we are splitting on the last "last action" (if it was repeated multiple times)
    assert len(log[-1].rsplit(action, 1)) <= 2
    # Case A — if not the entire reasoning trace, meaning action is embedded in a longer message
    if action != log[-1].replace('<|eot_id|>', ''):
      log_last: str = log[-1].rsplit(action, 1)
      # remove the newlines right before and after action
      # (adding the newline right in front of action so it's possible to extract the action)
      # \n seems to be independent from <
      # but \n sticks to the end of >, sp >\n\n is a token
      log_last = log_last[0].rstrip() + '\n' + action + '\n\n' + log_last[1].lstrip()
    # Case B — Action is the whole message
    else:
      log_last = log[-1].replace(action, action + '\n\n') # Still appends \n\n so tokenization is consistent
    
    # Rebuild the full reasoning trace
    log_last = log_last.replace('<|eot_id|>', '')
    log_except_last = '\n\n'.join(log[:-1]).replace('<|eot_id|>', '')
    reasoning_trace = log_except_last + '\n\n' + log_last
    assert '<|eot_id|>' not in reasoning_trace

    kv = get_log_kv(reasoning_trace, tokenizer, model, emb) # Encode full reasoning trace into KV
    # At this point: KV contains all tokens

    # find the token start and end_idx corresponding to the last action
    input_ids_action = tokenizer.encode(action + '\n\n', return_tensors='pt', add_special_tokens=False) # action + \n\n
    input_ids_last = tokenizer.encode(log_last, return_tensors='pt', add_special_tokens=False) # last message
    input_ids_except_last = tokenizer.encode(log_except_last + '\n\n', return_tensors='pt', add_special_tokens=False) # all except last message

    start_idx, end_idx = sublist_indices(input_ids_last[0], input_ids_action[0]) # find the start and end_idx in the last message
    
    # from input_ids_last, find the start and end_idx corresponding to input_ids_action
    # then add it to the start_idx of input_ids_last
    last_start_idx = input_ids_except_last.shape[1] # the start index of the last message in the full reasoning trace (execpt last message)
    start_idx += last_start_idx # adjust to full reasoning trace
    end_idx += last_start_idx # adjust to full reasoning trace

    input_ids_reasoning_trace = tokenizer.encode(reasoning_trace, return_tensors='pt', add_special_tokens=False) # full reasoning trace
    assert tokenizer.decode(input_ids_reasoning_trace[0][start_idx:end_idx]) == action + '\n\n' 
    assert '<|eot_id|>' not in action

    kv._seen_tokens = end_idx - start_idx # update the number of seen tokens to only account for the last action (because of slicing below)
    num_layers = len(kv)
    for layer_idx in range(num_layers):
      # for every layer, we throw away the KV values of all the tokens before and after the last action and only keep the tokens of the last action
      # slicing only return view, not changing the original tensor
      # pickle saves the underlying storage
      kv.key_cache[layer_idx] = kv.key_cache[layer_idx][:, :, start_idx:end_idx, :].clone().contiguous() 
      kv.value_cache[layer_idx] = kv.value_cache[layer_idx][:, :, start_idx:end_idx, :].clone().contiguous()
      assert kv.key_cache[layer_idx].shape == kv.value_cache[layer_idx].shape
      assert kv.key_cache[layer_idx].shape[2] == kv._seen_tokens == input_ids_action.shape[1]

    kv.key_cache = [x.cpu() for x in kv.key_cache]
    kv.value_cache = [x.cpu() for x in kv.value_cache]
    log_kvs.append(kv) 
  
  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_action.pkl') # save the KV caches to disk as pickle files


# ---------------------------------------------------------------------------
# --------------------------- NEW IMPLEMENTATION ----------------------------
# ---------------------------------------------------------------------------

def store_reasoning_last_k_actions(dataset: str, k: int):
    '''
    Encode the entire reasoning trace, but store the KV values corresponding
    to the last k agentic actions in the reasoning trace.
    '''

    model_id = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    config: LlamaConfig = AutoConfig.from_pretrained(model_id)
    emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(
        device=model.device,
        dtype=torch.float32
    )

    logs = read_json(f'./data/{dataset}/preds/log.json')
    log_kvs = []

    for log in tqdm(logs):
        # keep only assistant messages
        assistant_logs = [x for i, x in enumerate(log[0]) if i % 2 == 1]

        # extract actions from each assistant message
        actions = [extract_action(x).replace('<|eot_id|>', '') for x in assistant_logs]

        # keep last k actions (skip empty ones just in case)
        actions = [a for a in actions if a.strip() != '']
        actions = actions[-k:]

        assert len(actions) > 0

        # rebuild reasoning trace with clean action placement
        rebuilt_logs = []
        action_spans = []  # (start_idx, end_idx) in token space

        for msg, action in zip(assistant_logs, actions):
            if action not in msg:
                continue

        # rebuild full reasoning trace
        reasoning_trace = '\n\n'.join(
            [x.replace('<|eot_id|>', '') for x in assistant_logs]
        )

        kv = get_log_kv(reasoning_trace, tokenizer, model, emb)

        # tokenize full trace once
        input_ids_trace = tokenizer.encode(
            reasoning_trace,
            return_tensors='pt',
            add_special_tokens=False
        )[0]

        kept_spans = []

        # find token spans for each of the last k actions
        for action in actions:
            action_ids = tokenizer.encode(
                action + '\n\n',
                return_tensors='pt',
                add_special_tokens=False
            )[0]

            start_idx, end_idx = sublist_indices(input_ids_trace, action_ids)
            kept_spans.append((start_idx, end_idx))

        # concatenate spans in order
        total_tokens = sum(end - start for start, end in kept_spans)
        kv._seen_tokens = total_tokens

        num_layers = len(kv)
        for layer_idx in range(num_layers):
            k_chunks = []
            v_chunks = []

            for start, end in kept_spans:
                k_chunks.append(
                    kv.key_cache[layer_idx][:, :, start:end, :]
                )
                v_chunks.append(
                    kv.value_cache[layer_idx][:, :, start:end, :]
                )

            kv.key_cache[layer_idx] = torch.cat(k_chunks, dim=2).clone().contiguous()
            kv.value_cache[layer_idx] = torch.cat(v_chunks, dim=2).clone().contiguous()

            assert kv.key_cache[layer_idx].shape[2] == kv._seen_tokens

        # move to CPU for storage
        kv.key_cache = [x.cpu() for x in kv.key_cache]
        kv.value_cache = [x.cpu() for x in kv.value_cache]

        log_kvs.append(kv)

    write_pickle(
        log_kvs,
        f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_{k}_actions.pkl'
    )






if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str)
  args = parser.parse_args()

  # choose one encode-storage strategy from below
  # default is store_reasoning_offline(args.dataset, last_num=1)
  store_reasoning_offline(args.dataset, last_num=1)
  # store_reasoning_last_as_context(args.dataset)
  # store_reasoning_all(args.dataset)
  # store_reasoning_last_action(args.dataset)