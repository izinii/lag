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

    # Find the exact tokens sequence corresponding to the last action inside the last assistant message
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
# ------------------------ NEW METHODS IMPLEMENTED --------------------------
# ---------------------------------------------------------------------------

# --- Store kv of last k rounds actions ---
def store_reasoning_last_k_actions(dataset: str, last_k: int):
  '''Encode all reasoning traces, store the KV values corresponding to the last k agentic actions in the reasoning trace'''
  # Instead of slicing KV for one [start_idx:end_idx] span, we slice KV for k spans and concatenate them (in order).

  model_id = 'meta-llama/Llama-3.1-8B-Instruct'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
  config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
  emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

  logs = read_json(f'./data/{dataset}/preds/log.json')
  log_kvs = []

  for log in tqdm(logs):
    log = [x for i, x in enumerate(log[0]) if i % 2 == 1]  # use only assistant messages
    actions = [extract_action(x).replace('<|eot_id|>', '') for x in log] # extract actions from all assistant messages

    # keep last k actions
    actions = [a for a in actions if a.strip() != '']
    actions = actions[-last_k:]
    assert len(actions) > 0

    log_last = log[-1].replace('<|eot_id|>', '') # rebuild last message so that all actions appear cleanly and once

    for action in actions:
      assert len(log_last.rsplit(action, 1)) <= 2
      if action in log_last:
        parts = log_last.rsplit(action, 1)
        log_last = (parts[0].rstrip() + '\n' + action + '\n\n' + parts[1].lstrip())     

    log_except_last = '\n\n'.join(log[:-1]).replace('<|eot_id|>', '')
    reasoning_trace = log_except_last + '\n\n' + log_last
    assert '<|eot_id|>' not in reasoning_trace

    kv = get_log_kv(reasoning_trace, tokenizer, model, emb) # encode full reasoning trace into KV
    input_ids_reasoning_trace = tokenizer.encode(reasoning_trace, return_tensors='pt', add_special_tokens=False)[0] # tokenize once for index resolution

    # find token spans for each action
    spans = []
    for action in actions:
      input_ids_action = tokenizer.encode(action + '\n\n', return_tensors='pt', add_special_tokens=False)[0]
      start_idx, end_idx = sublist_indices(input_ids_reasoning_trace, input_ids_action) # find the start and end_idx in the full reasoning trace for a specific action
      spans.append((start_idx, end_idx)) # keep the span

    kv._seen_tokens = sum(end - start for start, end in spans)  # total number of kept tokens
    num_layers = len(kv)
    for layer_idx in range(num_layers):
      key_chunks = []
      value_chunks = []

      for start_idx, end_idx in spans:
        key_chunks.append(kv.key_cache[layer_idx][:, :, start_idx:end_idx, :])
        value_chunks.append(kv.value_cache[layer_idx][:, :, start_idx:end_idx, :])

      kv.key_cache[layer_idx] = torch.cat(key_chunks, dim=2).clone().contiguous()
      kv.value_cache[layer_idx] = torch.cat(value_chunks, dim=2).clone().contiguous()
      assert kv.key_cache[layer_idx].shape == kv.value_cache[layer_idx].shape
      assert kv.key_cache[layer_idx].shape[2] == kv._seen_tokens

    # move to CPU for storage
    kv.key_cache = [x.cpu() for x in kv.key_cache]
    kv.value_cache = [x.cpu() for x in kv.value_cache]

    log_kvs.append(kv)

  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_{last_k}_actions.pkl') # save the KV caches to disk as pickle files


# --- Get the last k rounds response and select S KVs randomly (per-layer) ---
def store_reasoning_offline_randomness(dataset, last_num: int, S: int):
  '''The default storage strategy: Encode all reasoning traces, store the KV values corresponding to the last_num reasoning trace
  (e.g., if last_num = 2, store the KV values corresponding to tokens in the last two reasoning traces)
  *** BUT randomly sample S keys/values from the last N responses per layer ***
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
    # At this point: kv.key_cache[layer].shape = (1, H, T, D) and T = number of tokens in last k responses

    # RANDOMLY sample S keys/values from the last N responses per layer
    total_tokens = kv._seen_tokens
    assert total_tokens == kv.key_cache[0].shape[2]
    if S < total_tokens:
      sampled_indices = torch.randperm(total_tokens)[:S].sort().values  # randomly sample S unique indices and sort them
      kv._seen_tokens = S
      num_layers = len(kv)
      for layer_idx in range(num_layers):
        kv.key_cache[layer_idx] = kv.key_cache[layer_idx][:, :, sampled_indices, :].clone().contiguous()
        kv.value_cache[layer_idx] = kv.value_cache[layer_idx][:, :, sampled_indices, :].clone().contiguous()
        assert kv.key_cache[layer_idx].shape == kv.value_cache[layer_idx].shape
        assert kv.key_cache[layer_idx].shape[2] == S
    else:
      # if S >= total_tokens, keep all tokens
      kv._seen_tokens = total_tokens # not mandatory, but just in case 

    log_kvs.append(kv)

  write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_{last_num}_randomness_{S}.pkl') # save the KV caches to disk as pickle files


# --- Get the last k rounds response and select the top-S KVs (per-layer) based on the attention to the last action ---
def store_reasoning_offline_topS(dataset: str, last_num: int, S: int):
    '''Encode all reasoning traces, keep KV of the last `last_num` rounds, then select top-S KV tokens per layer based on attention to the last action.'''

    model_id = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)
    config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
    emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)

    logs = read_json(f'./data/{dataset}/preds/log.json')
    log_kvs = []

    for log in tqdm(logs):
        log = [x for i, x in enumerate(log[0]) if i % 2 == 1] # use only assistant messages
        action = extract_action(log[-1]).replace('<|eot_id|>', '') # extract last action
        
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

        input_ids = tokenizer.encode(reasoning_trace, return_tensors='pt', add_special_tokens=False).to(model.device) # tokenize for attention computation

        # Forward pass WITH attentions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True, output_attentions=True)
        kv = outputs.past_key_values # shape: (1, heads, T, head_dim)
        attentions = outputs.attentions  # shape: (1, heads, T, T)

         # find the token start and end_idx corresponding to the last action
        input_ids_action = tokenizer.encode(action + '\n\n', return_tensors='pt', add_special_tokens=False) # action + \n\n
        input_ids_last = tokenizer.encode(log_last, return_tensors='pt', add_special_tokens=False) # last message
        input_ids_except_last = tokenizer.encode(log_except_last + '\n\n', return_tensors='pt', add_special_tokens=False) # all except last message

        # Find the exact tokens sequence corresponding to the last action inside the last assistant message
        action_start, action_end = sublist_indices(input_ids_last[0], input_ids_action[0]) # find the start and end_idx in the last message

        # from input_ids_last, find the start and end_idx corresponding to input_ids_action
        # then add it to the start_idx of input_ids_last
        action_start += input_ids_except_last.shape[0]
        action_end   += input_ids_except_last.shape[0]

        # last-k responses span
        assistant_logs = [x for i, x in enumerate(log) if i % 2 == 1]
        last_k_text = '\n\n'.join(assistant_logs[-last_num:]).replace('<|eot_id|>', '') + '\n\n'
        input_ids_last_k = tokenizer.encode(last_k_text, return_tensors='pt', add_special_tokens=False)[0]
        last_k_start = input_ids.shape[1] - input_ids_last_k.shape[0]
        last_k_end   = input_ids.shape[1]


        num_layers = len(kv)
        new_kv = []
        for layer_idx in range(num_layers):
            attn = attentions[layer_idx][0]  # (H, T, T)

            # restrict attention: action → last-k
            attn_slice = attn[:, action_start:action_end, last_k_start:last_k_end]  # (H, A, K)

            # aggregate → one score per last-k token
            scores = attn_slice.mean(dim=0).mean(dim=0)  # (K,)

            K = scores.shape[0]
            if S < K:
                top_idx = torch.topk(scores, S).indices
                top_idx, _ = torch.sort(top_idx)
            else:
                top_idx = torch.arange(K)

            global_idx = top_idx + last_k_start

            k_layer, v_layer = kv[layer_idx]
            k_sel = k_layer[:, :, global_idx, :].clone().contiguous()
            v_sel = v_layer[:, :, global_idx, :].clone().contiguous()

            new_kv.append((k_sel.cpu(), v_sel.cpu()))

        # Build DynamicCache-compatible object
        kv_out = type(kv)()
        kv_out.key_cache   = [k for k, _ in new_kv]
        kv_out.value_cache = [v for _, v in new_kv]
        kv_out._seen_tokens = len(new_kv[0][0][0, 0])

        log_kvs.append(kv_out)

    write_pickle(log_kvs, f'{STORE_PREFIX}/kv/{dataset}/reasoning_last_{last_num}_topS_{S}.pkl')



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str)
  args = parser.parse_args()

  # choose one encode-storage strategy from below :
  store_reasoning_offline(args.dataset, last_num=1) # default method
  # store_reasoning_last_as_context(args.dataset)
  # store_reasoning_all(args.dataset)
  # store_reasoning_last_action(args.dataset)

  # new implemented methods:
  # store_reasoning_last_k_actions(args.dataset, last_k=2)
  # store_reasoning_offline_randomness(args.dataset, last_num=2, S=64)
  # store_reasoning_offline_topS(args.dataset, last_num=2, S=64)