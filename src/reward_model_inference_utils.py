import numpy as np
import torch
import logging
import random
from tqdm import tqdm

def calculate_accuracy(chosen_scores, rejected_scores):
    """Calculate accuracy based on chosen and rejected scores."""
    chosen_scores_np = np.array(chosen_scores)
    rejected_scores_np = np.array(rejected_scores)
    
    comparison = chosen_scores_np > rejected_scores_np
    
    accuracy = np.mean(comparison) * 100
    correct_indices = np.where(comparison)[0]
    
    return accuracy, len(correct_indices), correct_indices

def run_redundancy_tests(chosen_scores, rejected_scores, redundant_indices, total_dataset_size):
    """Run redundancy tests using pre-computed scores instead of re-running the model."""
    
    # Calculate accuracy and correct predictions for full dataset
    full_accuracy, _, correct_indices = calculate_accuracy(chosen_scores, rejected_scores)
    
    # 1. Test excluding redundant examples
    non_redundant_indices = [i for i in range(total_dataset_size) if i not in redundant_indices]
    chosen_non_redundant = [chosen_scores[i] for i in non_redundant_indices]
    rejected_non_redundant = [rejected_scores[i] for i in non_redundant_indices]
    accuracy_non_redundant, correct_count_non_redundant, _ = calculate_accuracy(
        chosen_non_redundant, rejected_non_redundant)
    
    # 2. Test only redundant examples
    chosen_redundant = [chosen_scores[i] for i in redundant_indices]
    rejected_redundant = [rejected_scores[i] for i in redundant_indices]
    accuracy_redundant, correct_count_redundant, _ = calculate_accuracy(
        chosen_redundant, rejected_redundant)
    
    # 3. Test random exclusion of same percentage
    redundant_percentage = len(redundant_indices) / total_dataset_size
    random_exclude_count = int(total_dataset_size * redundant_percentage)
    
    random.seed(42)  # For reproducibility
    random_exclude_indices = random.sample(range(total_dataset_size), random_exclude_count)
    random_keep_indices = [i for i in range(total_dataset_size) if i not in random_exclude_indices]
    
    chosen_random = [chosen_scores[i] for i in random_keep_indices]
    rejected_random = [rejected_scores[i] for i in random_keep_indices]
    accuracy_random, correct_count_random, _ = calculate_accuracy(
        chosen_random, rejected_random)
    
    return {
        "accuracy_full": full_accuracy,
        "accuracy_non_redundant": accuracy_non_redundant,
        "accuracy_redundant": accuracy_redundant,
        "accuracy_random": accuracy_random,
        "counts": {
            "total": total_dataset_size,
            "redundant": len(redundant_indices),
            "non_redundant": len(non_redundant_indices),
            "random_excluded": random_exclude_count
        }
    }



def process_examples_gemma(model, tokenizer, dataset, device, args):
    """Process dataset examples and compute features."""
    
    features_chosen = []
    features_chosen_full_length = []

    features_rejected = []
    features_rejected_full_length = []
    features_diff = []
    chosen_scores = []
    rejected_scores = []

    for example in tqdm(dataset, desc="Processing examples"):
        prompt = example['prompt']
        chosen_completion = example['chosen']
        rejected_completion = example['rejected']

        conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_completion}]
        conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_completion}]

        conv1_tokenized = tokenizer.apply_chat_template(conv1, return_tensors="pt").to(device)
        conv2_tokenized = tokenizer.apply_chat_template(conv2, return_tensors="pt").to(device)

        with torch.no_grad():
            # with torch.amp.autocast('cuda'):
                output_1 = model(conv1_tokenized, output_hidden_states=True)
                output_2 = model(conv2_tokenized, output_hidden_states=True)


                score_chosen = output_1.score.cpu().float() 
                score_rejected = output_2.score.cpu().float() 
                chosen_scores.append(score_chosen)
                rejected_scores.append(score_rejected)

                # try:
                #     print(output_1.keys())
                # except Exception as e:
                #     print(e)

                # try:
                #     print(output_1)
                # except Exception as e:
                #     print(e)

                if args.using_peft:
                    hidden_states1 = output_1.hidden_states[-1][:, -1, :]
                    hidden_states2 = output_2.hidden_states[-1][:, -1, :]

                    cls_embedding1 = model.score[0](hidden_states1).cpu().squeeze()
                    cls_embedding2 = model.score[0](hidden_states2).cpu().squeeze()

                else:  

                    output_1_gemma_base = model.model(conv1_tokenized, output_hidden_states=True)
                    output_2_gemma_base = model.model(conv2_tokenized, output_hidden_states=True)

                    hidden_states1 = output_1_gemma_base.hidden_states
                    hidden_states2 = output_2_gemma_base.hidden_states
                    # Get the last token embedding that isn't a PAD

                    cls_embedding1 = hidden_states1[-1][:, -1, :].cpu().squeeze()
                    cls_embedding2 = hidden_states2[-1][:, -1, :].cpu().squeeze()

                
                if args.shorten_size:
                    features_chosen.append(cls_embedding1[:int(args.shorten_size)].cpu())
                    features_rejected.append(cls_embedding2[:int(args.shorten_size)].cpu())

                    features_chosen_full_length.append(cls_embedding1[:int(args.shorten_size)].cpu())
                    features_rejected_full_length.append(cls_embedding2[:int(args.shorten_size)].cpu())  
                    features_diff.append((cls_embedding1[:int(args.shorten_size)] - cls_embedding2[:int(args.shorten_size)]).cpu())
                else:
                    features_chosen.append(cls_embedding1.cpu())
                    features_rejected.append(cls_embedding2.cpu())

    return torch.stack(features_chosen), torch.stack(features_rejected), \
            torch.stack(features_chosen_full_length),  torch.stack(features_rejected_full_length), \
            torch.stack(features_diff), chosen_scores, rejected_scores


def process_examples(model, tokenizer, dataset, device, args):
    """Process dataset examples and compute features."""
    
    print(model)
    features_chosen = []
    features_chosen_full_length = []

    features_rejected = []
    features_rejected_full_length = []
    features_diff = []
    chosen_scores = []
    rejected_scores = []

    for example in tqdm(dataset, desc="Processing examples"):
        prompt = example['prompt']
        chosen_completion = example['chosen']
        rejected_completion = example['rejected']

        conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen_completion}]
        conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected_completion}]

        tokenizer.padding_side = "left"
        conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
        conv2_formatted = tokenizer.apply_chat_template(conv2, tokenize=False)
        conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt", padding=True, truncation=False).to(device)
        # .to(torch.bfloat16)
        conv2_tokenized = tokenizer(conv2_formatted, return_tensors="pt",  padding=True, truncation=False).to(device)
        # .to(torch.bfloat16)
        model.eval()
        with torch.no_grad():
            # with torch.amp.autocast('cuda'): #casting causes issues with BF16
                output_1 = model(**conv1_tokenized, output_hidden_states=True)
                output_2 = model(**conv2_tokenized, output_hidden_states=True)

                score_chosen = output_1.logits[0][0].item()
                score_rejected = output_2.logits[0][0].item()

                chosen_scores.append(score_chosen)
                rejected_scores.append(score_rejected)


                if args.using_peft:
                    hidden_states1 = output_1.hidden_states[-1][:, -1, :]
                    hidden_states2 = output_2.hidden_states[-1][:, -1, :]

                    cls_embedding1 = model.score[0](hidden_states1).cpu().squeeze()
                    cls_embedding2 = model.score[0](hidden_states2).cpu().squeeze()
                else:
                    hidden_states1 = output_1.hidden_states
                    hidden_states2 = output_2.hidden_states
                    # Get the last token embedding that isn't a PAD
                    cls_embedding1 = hidden_states1[-1][:, -1, :].cpu().squeeze()
                    cls_embedding2 = hidden_states2[-1][:, -1, :].cpu().squeeze()

                if args.shorten_size:
                    features_chosen.append(cls_embedding1[:int(args.shorten_size)].cpu())
                    features_rejected.append(cls_embedding2[:int(args.shorten_size)].cpu())

                    features_chosen_full_length.append(cls_embedding1[:int(args.shorten_size)].cpu())
                    features_rejected_full_length.append(cls_embedding2[:int(args.shorten_size)].cpu())  
                    features_diff.append((cls_embedding1[:int(args.shorten_size)] - cls_embedding2[:int(args.shorten_size)]).cpu())
                else:
                    features_chosen.append(cls_embedding1.cpu())
                    features_rejected.append(cls_embedding2.cpu())

    return torch.stack(features_chosen), torch.stack(features_rejected), \
            torch.stack(features_chosen_full_length),  torch.stack(features_rejected_full_length), \
            torch.stack(features_diff), chosen_scores, rejected_scores


