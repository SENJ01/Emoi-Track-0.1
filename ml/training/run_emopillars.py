import argparse
import json
import logging
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW

from ml.models.model import RobertaForMultiLabelClassification

from ml.utils.utils import (
    init_logger,
    set_seed,
    compute_metrics
)

from ml.data.data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)

logger = logging.getLogger(__name__)

# =====================================================
# EVALUATION FUNCTION
# =====================================================
def evaluate(args, model, dataset):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.eval_batch_size
    )

    model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }

            outputs = model(**inputs)
            loss, logits = outputs[:2]

        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        labels = inputs["labels"].cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    avg_loss = total_loss / len(dataloader)

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    hard_preds = (all_probs >= args.threshold).astype(int)
    metrics = compute_metrics(all_labels, hard_preds)

    return avg_loss, metrics, all_probs, all_labels

# =====================================================
# TRAINING FUNCTION
# =====================================================
def train(args, model, tokenizer, train_dataset, dev_dataset):
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size
    )

    total_steps = len(train_loader) * args.num_train_epochs

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_proportion),
        num_training_steps=total_steps
    )

    global_step = 0
    model.zero_grad()

    running_loss = 0.0
    train_probs_buffer = []
    train_labels_buffer = []

    for epoch in range(args.num_train_epochs):
        model.train()

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        ):
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }

            outputs = model(**inputs)
            loss, logits = outputs[:2]

            running_loss += loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()

            train_probs_buffer.append(probs)
            train_labels_buffer.append(labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm
            )

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1

            # ==================================================
            # SAVE PER CHECKPOINT
            # ==================================================
            if global_step % args.save_steps == 0:
                avg_train_loss = running_loss / args.save_steps

                train_probs = np.vstack(train_probs_buffer)
                train_labels = np.vstack(train_labels_buffer)

                hard_train_preds = (train_probs >= args.threshold).astype(int)
                train_metrics = compute_metrics(
                    train_labels,
                    hard_train_preds
                )

                train_accuracy = train_metrics["accuracy"]

                running_loss = 0.0
                train_probs_buffer = []
                train_labels_buffer = []

                checkpoint_dir = os.path.join(
                    args.output_dir,
                    f"checkpoint-{global_step}"
                )
                os.makedirs(checkpoint_dir, exist_ok=True)

                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

                dev_loss, dev_metrics, _, _ = evaluate(
                    args, model, dev_dataset
                )

                dev_dir = os.path.join(args.output_dir, "dev")
                os.makedirs(dev_dir, exist_ok=True)

                dev_file = os.path.join(
                    dev_dir,
                    f"dev-{global_step}.txt"
                )

                with open(dev_file, "w", encoding="utf-8") as f:
                    f.write("----- TRAIN METRICS -----\n")
                    f.write(f"train_loss = {avg_train_loss}\n")
                    f.write(f"train_accuracy = {train_accuracy}\n\n")

                    f.write("----- VALIDATION METRICS -----\n")
                    f.write(f"val_loss = {dev_loss}\n")
                    f.write(f"val_accuracy = {dev_metrics['accuracy']}\n\n")

                    f.write(f"macro_f1 = {dev_metrics['macro_f1']}\n")
                    f.write(f"macro_precision = {dev_metrics['macro_precision']}\n")
                    f.write(f"macro_recall = {dev_metrics['macro_recall']}\n")
                    f.write(f"micro_f1 = {dev_metrics['micro_f1']}\n")
                    f.write(f"micro_precision = {dev_metrics['micro_precision']}\n")
                    f.write(f"micro_recall = {dev_metrics['micro_recall']}\n")
                    f.write(f"weighted_f1 = {dev_metrics['weighted_f1']}\n")
                    f.write(f"weighted_precision = {dev_metrics['weighted_precision']}\n")
                    f.write(f"weighted_recall = {dev_metrics['weighted_recall']}\n")

                logger.info("Checkpoint %s saved.", global_step)

    return global_step

# =====================================================
# MAIN
# =====================================================
def main(cli_args):
    config_path = os.path.join(
        "ml", "config", f"{cli_args.taxonomy}.json"
    )

    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)

    args = type("Args", (), config_dict)

    if not hasattr(args, "hf_revision"):
        args.hf_revision = "main"

    os.makedirs(args.output_dir, exist_ok=True)

    init_logger()
    set_seed(args)

    processor = GoEmotionsProcessor(args)
    label_list = processor.get_labels()

    # ---------------- MODEL ----------------
    if args.model_type.lower() == "roberta":
        config = RobertaConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.hf_revision,
            num_labels=len(label_list)
        )

        tokenizer = RobertaTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            revision=args.hf_revision
        )

        model = RobertaForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            revision=args.hf_revision,
            config=config
        )

    else:
        raise ValueError("Unsupported model type")

    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    model.to(args.device)

    # ---------------- DATA ----------------
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    # ---------------- TRAIN ----------------
    train(args, model, tokenizer, train_dataset, dev_dataset)

    # ---------------- FINAL TEST ----------------
    logger.info("Running final test evaluation...")

    test_loss, test_metrics, raw_probs, labels = evaluate(
        args, model, test_dataset
    )

    test_dir = os.path.join(args.output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    np.save(os.path.join(test_dir, "raw_probs.npy"), raw_probs)
    np.save(os.path.join(test_dir, "labels.npy"), labels)

    with open(os.path.join(test_dir, "test-base.txt"), "w", encoding="utf-8") as f:
        f.write(f"accuracy = {test_metrics['accuracy']}\n")
        f.write(f"loss = {test_loss}\n")
        f.write(f"macro_f1 = {test_metrics['macro_f1']}\n")
        f.write(f"macro_precision = {test_metrics['macro_precision']}\n")
        f.write(f"macro_recall = {test_metrics['macro_recall']}\n")
        f.write(f"micro_f1 = {test_metrics['micro_f1']}\n")
        f.write(f"micro_precision = {test_metrics['micro_precision']}\n")
        f.write(f"micro_recall = {test_metrics['micro_recall']}\n")
        f.write(f"weighted_f1 = {test_metrics['weighted_f1']}\n")
        f.write(f"weighted_precision = {test_metrics['weighted_precision']}\n")
        f.write(f"weighted_recall = {test_metrics['weighted_recall']}\n")

    logger.info("Test evaluation complete.")

# =====================================================
# CLI
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=str, required=True)
    cli_args = parser.parse_args()
    main(cli_args)