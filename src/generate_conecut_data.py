import logging
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

subset_mapping = {
    "alpacaeval-easy": "chat",
    "alpacaeval-length": "chat",
    "alpacaeval-hard": "chat",
    "mt-bench-easy": "chat",
    "mt-bench-med": "chat",

    "mt-bench-hard": "chat_hard",
    "llmbar-natural": "chat_hard",
    "llmbar-adver-neighbor": "chat_hard",
    "llmbar-adver-GPTInst": "chat_hard",
    "llmbar-adver-GPTOut": "chat_hard",
    "llmbar-adver-manual": "chat_hard",

    "refusals-dangerous": "safety",
    "refusals-offensive": "safety",
    "xstest-should-refuse": "safety",
    "xstest-should-respond": "safety",
    "donotanswer": "safety",

    "math-prm": "reasoning",
    "hep-cpp": "reasoning",
    "hep-go": "reasoning",
    "hep-java": "reasoning",
    "hep-js": "reasoning",
    "hep-python": "reasoning",
    "hep-rust": "reasoning"
}

def apply_chat_template(prompt, response, tokenizer):
    templated_example = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    templated_example = tokenizer.apply_chat_template(templated_example, tokenize=False)
    templated_example = templated_example[len(tokenizer.bos_token):]

    return templated_example

def run_inference(examples, model, tokenizer):
    dataloader = DataLoader(
        examples,
        batch_size=16,
        collate_fn=lambda batch: tokenizer(batch, return_tensors="pt", padding=True),
    )

    cached_activations = {}
    def score_hook(module, inputs, output):
        cached_activations["score_inputs"] = inputs[0]
    score_hook_handle = model.score.register_forward_hook(score_hook)

    scores_batches = []
    activations_batches = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                logger.info(f"Processing batch {i + 1}/{len(dataloader)}...")

            batch = {key: value.to("cuda:0") for key, value in batch.items()}
            out = model(**batch, return_dict=True)

            scores_batch = out.logits.squeeze(-1).cpu()
            scores_batches.append(scores_batch)

            score_inputs = cached_activations["score_inputs"]
            example_indices = torch.arange(score_inputs.size(0), device="cuda:0")
            final_token_indices = batch["attention_mask"].sum(dim=1) - 1
            activations_batch = score_inputs[example_indices, final_token_indices, :].cpu()
            activations_batches.append(activations_batch)
            cached_activations["score_inputs"] = None
    
    score_hook_handle.remove()

    return torch.cat(scores_batches, dim=0), torch.cat(activations_batches, dim=0)

def generate_conecut_data(arguments):
    logger.info("Loading dataset, model, and tokenizer...")
    dataset = load_dataset("allenai/reward-bench", split="filtered")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
        num_labels=1,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Llama-3.1-8B")
    logger.info("Completed loading dataset, model, and tokenizer.")

    if arguments.subset_filter != "full":
        logger.info("Filtering dataset using subset filter...")
        dataset = dataset.filter(lambda example: subset_mapping[example["subset"]] == arguments.subset_filter)
        logger.info("Completed filtering dataset using subset filter.")

    logger.info("Applying chat template to dataset...")
    templated_chosen_examples = []
    templated_rejected_examples = []
    for example in dataset:
        templated_chosen_examples.append(apply_chat_template(example["prompt"], example["chosen"], tokenizer))
        templated_rejected_examples.append(apply_chat_template(example["prompt"], example["rejected"], tokenizer))
    logger.info("Completed applying chat template to dataset.")

    logger.info("Running inference on the templated examples...")
    chosen_scores, chosen_activations = run_inference(templated_chosen_examples, model, tokenizer)
    rejected_scores, rejected_activations = run_inference(templated_rejected_examples, model, tokenizer)
    logger.info("Completed running inference on the templated examples.")

    logger.info("Extracting example subset labels...")
    subsets = [example["subset"] for example in dataset]
    logger.info("Completed extracting example subset labels.")

    return chosen_activations - rejected_activations, chosen_scores, rejected_scores, subsets
