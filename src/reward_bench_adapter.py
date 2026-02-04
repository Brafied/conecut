from rewardbench import load_eval_dataset, REWARD_MODEL_CONFIG, check_tokenizer_chat_template
from rewardbench.constants import SUBSET_MAPPING
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch, math, random, inspect
from fastchat.conversation import get_conv_template
from rewardbench.chattemplates import *  # This registers the chat templates

def build_rm_pipeline(model_id: str,
                      torch_dtype=torch.bfloat16,
                      quantize_toggle: bool | None = None,
                      attn_impl: str | None = None,
                      cache_dir: str | None = None):
    """Build reward model pipeline using RewardBench's infrastructure."""
    
    cfg = REWARD_MODEL_CONFIG.get(model_id, REWARD_MODEL_CONFIG["default"])
    model_builder = cfg["model_builder"]
    pipeline_builder = cfg["pipeline_builder"]
    quantized_default = cfg["quantized"]
    model_kwargs = {}

    if quantize_toggle is None:
        quantized = quantized_default
    else:
        quantized = quantize_toggle

    # auto-disable quant for bfloat16 (mimic run_rm.py)
    if torch_dtype == torch.bfloat16:
        quantized = False

    if quantized:
        model_kwargs.update(
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    else:
        model_kwargs.update(
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    # Build model and tokenizer
    if 'ldl' in model_id.lower():
        import os
        if cache_dir:
            # Create model-specific cache path
            model_cache_path = os.path.join(cache_dir, model_id.replace('/', '--'))
            if os.path.exists(model_cache_path):
                # Load from cache if exists
                model = model_builder(model_cache_path, **model_kwargs, trust_remote_code=True)
            else:
                # Download to cache directory
                from huggingface_hub import snapshot_download
                os.makedirs(cache_dir, exist_ok=True)
                model_cache_path = snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_dir=model_cache_path)
                model = model_builder(model_cache_path, **model_kwargs, trust_remote_code=True)
        else:
            model = model_builder(model_id, **model_kwargs, trust_remote_code=True)
    else:
        model = model_builder(model_id, **model_kwargs, trust_remote_code=True, cache_dir=cache_dir)
    
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    
    # Build pipeline using RewardBench's system
    reward_pipe = pipeline_builder(
        "text-classification",
        model=model,
        tokenizer=tok,
    )
    
    return tok, model, reward_pipe

def get_chat_template_for_model(model_id: str):
    """Get appropriate chat template for a model."""
    # Check if model requires custom dialogue formatting
    cfg = REWARD_MODEL_CONFIG.get(model_id, REWARD_MODEL_CONFIG["default"])
    custom_dialogue = cfg.get("custom_dialogue", False)
    
    if custom_dialogue:
        return None  # Custom dialogue formatting models handle their own templates
    
    # For openbmb models, use the openbmb chat template
    if "openbmb" in model_id.lower():
        try:
            return get_conv_template("openbmb")
        except Exception:
            # Fallback to default if template not found
            return None
    
    # For other models, let the tokenizer handle it if it has a template
    return None

def load_subset(tokenizer,
                model_id: str,
                section: str | None = None,
                keep_indices: list[int] | None = None):
    # Get appropriate chat template for this model
    conv_template = get_chat_template_for_model(model_id)
    
    # Check if model requires custom dialogue formatting
    cfg = REWARD_MODEL_CONFIG.get(model_id, REWARD_MODEL_CONFIG["default"])
    custom_dialogue_formatting = cfg.get("custom_dialogue", False)
    
    ds, subsets = load_eval_dataset(
        core_set=True,
        custom_dialogue_formatting=custom_dialogue_formatting,
        conv=conv_template,
        tokenizer=tokenizer,
        logger=None,
        return_extra_data=False,
    )

    if section:
        wanted = set(SUBSET_MAPPING[section])
        mask   = [i for i,s in enumerate(subsets) if s in wanted]
        ds, subsets = ds.select(mask), [subsets[i] for i in mask]

    if keep_indices is not None:
        ds, subsets = ds.select(keep_indices), [subsets[i] for i in keep_indices]

    return ds, subsets

def forward_collect_with_pipeline(reward_pipe, texts, max_length=2048):
    """Use RewardBench's pipeline system to get scores and extract features."""
    
    # Configure pipeline parameters like run_rm.py
    reward_pipeline_kwargs = {
        "truncation": True,
        "padding": True,
        "max_length": max_length,
    }
    
    print(f"Processing {len(texts)} texts with pipeline")
    print(f"First text sample: {texts[0][:100] if texts else 'No texts'}...")
    
    # Use RewardBench pipeline to get scores
    with torch.no_grad():
        raw_outputs = reward_pipe(texts, **reward_pipeline_kwargs)
    
    print(f"Raw pipeline outputs type: {type(raw_outputs)}")
    print(f"Raw pipeline outputs sample: {raw_outputs[:2] if isinstance(raw_outputs, list) else raw_outputs}")
    
    # Extract scores - handle RewardBench's different output formats
    scores = None
    if isinstance(raw_outputs, list) and len(raw_outputs) > 0:
        # Handle HF pipeline format: [{'label': 'LABEL_1', 'score': 0.68}, ...]
        if isinstance(raw_outputs[0], dict):
            if "score" in raw_outputs[0]:
                scores = torch.tensor([result["score"] for result in raw_outputs])
            elif "LABEL_1" in str(raw_outputs[0]):
                # Handle different dict format
                scores = torch.tensor([result.get("score", 0.0) for result in raw_outputs])
        else:
            # Handle tensor or float outputs
            scores = torch.tensor(raw_outputs)
    elif isinstance(raw_outputs, torch.Tensor):
        # Handle tensor outputs
        scores = raw_outputs.float().squeeze(-1).cpu()
    else:
        print(f"Warning: Unexpected output format: {type(raw_outputs)}")
        scores = torch.zeros(len(texts))
    
    if scores is None:
        print("Warning: Could not extract scores, using zeros")
        scores = torch.zeros(len(texts))
    
    print(f"Extracted scores shape: {scores.shape}")
    print(f"Extracted scores sample: {scores[:5]}")
    
    # Extract features by directly calling the model
    tokenizer = reward_pipe.tokenizer
    model = reward_pipe.model
    
    # Tokenize inputs for feature extraction
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs with hidden states
        outputs = model(**inputs, output_hidden_states=True)
    
    features = extract_features(model, outputs, inputs)
    
    print(f"Extracted features shape: {features.shape}")
    
    return scores, features

def extract_features(model, model_output, inputs):
    """Extract features based on model architecture."""
    model_name = type(model).__name__
    
    if hasattr(model_output, 'hidden_state') and model_output.hidden_state is not None:
        return model_output.hidden_state.cpu()
    
    if hasattr(model_output, 'hidden_states') and model_output.hidden_states is not None:
        hidden_states = model_output.hidden_states[-1]
        
        if 'DebertaV2' in model_name or 'Deberta' in model_name:
            # CLS based model
            return hidden_states[:, 0, :].cpu()
        elif any(arch in model_name for arch in ['Llama', 'Gemma', 'Qwen', 'Mistral']):
            # any of causal models, that use last token
            return hidden_states[:, -1, :].cpu()
        elif 'T5' in model_name:
            # T5 uses last encoder state
            return hidden_states.mean(dim=1).cpu()
        else:
            # Default: Use last token for most sequence classifiers
            return hidden_states[:, -1, :].cpu()
    
    if hasattr(model, 'model') and hasattr(model.model, 'hidden_states'):
        return model.model.hidden_states[-1][:, -1, :].cpu()
    
    raise ValueError(f"Could not extract features from model: {model_name}")

def run_rm_subset(model_id: str,
                  section: str | None = None,
                  keep_indices: list[int] | None = None,
                  batch_size: int = 8,
                  torch_dtype=torch.bfloat16,
                  max_length: int = 2048,
                  cache_dir: str | None = None):

    # Build pipeline using RewardBench's infrastructure
    tok, rm, reward_pipe = build_rm_pipeline(model_id, torch_dtype=torch_dtype, cache_dir=cache_dir)

    # Apply RewardBench's tokenization settings from run_rm.py
    tok.padding_side = "left"

    tok.truncation_side = "left"
    if not check_tokenizer_chat_template(tok):
        tok.add_eos_token = True
    if rm.config.pad_token_id is None:
        rm.config.pad_token_id = tok.eos_token_id
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    ds, subsets = load_subset(tok, model_id, section, keep_indices)

    scores_ch, scores_rj = [], []
    hid_ch, hid_rj       = [], []

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{math.ceil(len(ds)/batch_size)}")
        
        try:
            # Use RewardBench's pipeline system instead of custom forward_collect
            logits_ch, hidden_ch = forward_collect_with_pipeline(
                reward_pipe, batch["text_chosen"], max_length)
            logits_rj, hidden_rj = forward_collect_with_pipeline(
                reward_pipe, batch["text_rejected"], max_length)

            # Validate scores before adding
            if torch.isnan(logits_ch).any() or torch.isnan(logits_rj).any():
                print(f"Warning: NaN scores detected in batch {i//batch_size + 1}")
                print(f"Chosen scores: {logits_ch}")
                print(f"Rejected scores: {logits_rj}")
                # Replace NaNs with zeros for now
                logits_ch = torch.nan_to_num(logits_ch, nan=0.0)
                logits_rj = torch.nan_to_num(logits_rj, nan=0.0)

            scores_ch.extend(logits_ch.tolist())
            scores_rj.extend(logits_rj.tolist())
            hid_ch.append(hidden_ch)
            hid_rj.append(hidden_rj)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            print(f"Batch size: {len(batch['text_chosen'])}")
            print(f"Sample text chosen: {batch['text_chosen'][0][:100] if batch['text_chosen'] else 'None'}...")
            raise

    hid_ch = torch.cat(hid_ch)
    hid_rj = torch.cat(hid_rj)

    return hid_ch, hid_rj, scores_ch, scores_rj, subsets
