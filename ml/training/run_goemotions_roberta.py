import os
import sys
import argparse
import json
import logging
import glob

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict
from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

# Add project root so local modules can be imported when running this file directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import (  # noqa: E402
    BertForMultiLabelClassification,
    DistilBertForMultiLabelClassification,
    RobertaForMultiLabelClassification,
)
from utils.utils import init_logger, set_seed, compute_metrics  # noqa: E402
from data.data_loader import load_and_cache_examples, GoEmotionsProcessor  # noqa: E402

logger = logging.getLogger(__name__)


def get_hf_load_kwargs(path_or_name, revision=None):
    """Use revision only for remote Hugging Face repos, not local paths."""
    if revision and not os.path.exists(path_or_name):
        return {"revision": revision}
    return {}


def build_label_maps(label_list):
    """Create label-index mappings once for model config."""
    id2label = {str(i): label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    return id2label, label2id


def load_model_components(args, label_list):
    """Load config, tokenizer, and model based on the selected backbone."""
    id2label, label2id = build_label_maps(label_list)

    model_kwargs = get_hf_load_kwargs(args.model_name_or_path, args.hf_revision)
    tokenizer_kwargs = get_hf_load_kwargs(
        args.tokenizer_name_or_path, args.hf_revision
    )

    model_type = args.model_type.lower()

    if model_type == "bert":
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label=id2label,
            label2id=label2id,
            **model_kwargs,
        )
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            **tokenizer_kwargs,
        )
        model_class = BertForMultiLabelClassification

    elif model_type == "distilbert":
        config = DistilBertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            **model_kwargs,
        )
        tokenizer = DistilBertTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            **tokenizer_kwargs,
        )
        model_class = DistilBertForMultiLabelClassification

    elif model_type == "roberta":
        config = RobertaConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
            **model_kwargs,
        )
        tokenizer = RobertaTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            **tokenizer_kwargs,
        )
        model_class = RobertaForMultiLabelClassification

    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Load pretrained weights
    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        **model_kwargs,
    )
    return tokenizer, model, model_class


def evaluate(args, model, eval_dataset, mode, global_step=None, save_results=True):
    """Run evaluation and compute loss + classification metrics."""
    results = {}

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
    )

    logger.info(
        "***** Running evaluation on %s dataset %s *****",
        mode,
        f"({global_step} step)" if global_step else "",
    )
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Eval batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc=f"Evaluating-{mode}"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # Prepare model inputs
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type.lower() == "bert":
                inputs["token_type_ids"] = batch[2]

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        # Convert logits to probabilities
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = inputs["labels"].detach().cpu().numpy()

        # Collect predictions across batches
        if preds is None:
            preds = probs
            out_label_ids = labels
        else:
            preds = np.append(preds, probs, axis=0)
            out_label_ids = np.append(out_label_ids, labels, axis=0)

    # Average loss across all evaluation batches
    eval_loss = eval_loss / max(nb_eval_steps, 1)

    # Convert probabilities to binary outputs
    hard_preds = (preds > args.threshold).astype(int)

    results = {"loss": eval_loss}
    results.update(compute_metrics(out_label_ids, hard_preds))

    if save_results:
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        output_eval_file = os.path.join(
            mode_dir,
            f"{mode}-{global_step}.txt" if global_step else f"{mode}.txt",
        )

        with open(output_eval_file, "w", encoding="utf-8") as f_w:
            logger.info("***** Eval results on %s dataset *****", mode)
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, results[key])
                f_w.write(f"{key} = {results[key]}\n")

    return results


def append_training_history_row(history_path, row):
    """Append one row of training history to CSV."""
    row_df = pd.DataFrame([row])
    row_df.to_csv(
        history_path,
        mode="a",
        header=not os.path.exists(history_path),
        index=False,
    )


def create_train_dataloader(train_dataset, batch_size):
    """Create shuffled dataloader for training."""
    train_sampler = RandomSampler(train_dataset)
    return DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


def create_optimizer_and_scheduler(args, model, train_dataloader):
    """Prepare optimizer and linear warmup scheduler."""
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Exclude bias and LayerNorm weights from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total,
    )

    return optimizer, scheduler, t_total


def collect_interval_metrics(interval_train_preds, interval_train_labels, threshold):
    """Compute training metrics for the current logging interval."""
    train_preds = np.vstack(interval_train_preds)
    train_labels = np.vstack(interval_train_labels)
    train_preds_bin = (train_preds > threshold).astype(int)
    return compute_metrics(train_labels, train_preds_bin)


def get_eval_dataset_and_mode(args, dev_dataset, test_dataset):
    """Choose whether to evaluate on dev or test during training."""
    if args.evaluate_test_during_training:
        return test_dataset, "test"
    return dev_dataset, "dev"


def build_training_history_row(
    global_step,
    epoch,
    avg_train_loss,
    train_metrics,
    eval_results,
    scheduler,
):
    """Build one structured training-history record."""
    return {
        "step": global_step,
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": train_metrics.get("accuracy", 0.0),
        "train_macro_f1": train_metrics.get("macro_f1", 0.0),
        "train_micro_f1": train_metrics.get("micro_f1", 0.0),
        "val_loss": eval_results.get("loss", 0.0),
        "val_accuracy": eval_results.get("accuracy", 0.0),
        "val_macro_f1": eval_results.get("macro_f1", 0.0),
        "val_micro_f1": eval_results.get("micro_f1", 0.0),
        "learning_rate": scheduler.get_last_lr()[0],
    }


def evaluate_and_record_history(
    args,
    model,
    dev_dataset,
    test_dataset,
    global_step,
    epoch,
    interval_loss,
    interval_steps,
    interval_train_preds,
    interval_train_labels,
    scheduler,
    history_path,
    leftover=False,
):
    """Run evaluation and save one history row for the current interval."""
    train_metrics = collect_interval_metrics(
        interval_train_preds,
        interval_train_labels,
        args.threshold,
    )
    avg_train_loss = interval_loss / max(interval_steps, 1)

    eval_dataset, eval_mode = get_eval_dataset_and_mode(args, dev_dataset, test_dataset)
    eval_results = evaluate(
        args,
        model,
        eval_dataset,
        eval_mode,
        global_step=global_step,
        save_results=True,
    )

    row = build_training_history_row(
        global_step,
        epoch,
        avg_train_loss,
        train_metrics,
        eval_results,
        scheduler,
    )
    append_training_history_row(history_path, row)

    if leftover:
        logger.info(
            "Appended final leftover training history at step %s to %s",
            global_step,
            history_path,
        )
    else:
        logger.info(
            "Appended training history at step %s to %s",
            global_step,
            history_path,
        )


def save_checkpoint(args, model, tokenizer, optimizer, scheduler, global_step):
    """Save model, tokenizer, and optional optimizer state."""
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))

    if args.save_optimizer:
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))


def prepare_batch_inputs(args, batch):
    """Prepare the input dictionary expected by the transformer model."""
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "labels": batch[3],
    }
    if args.model_type.lower() == "bert":
        inputs["token_type_ids"] = batch[2]
    return inputs


def train(args, model, tokenizer, train_dataset, dev_dataset=None, test_dataset=None):
    """Main training loop with periodic evaluation and checkpointing."""
    train_dataloader = create_train_dataloader(train_dataset, args.train_batch_size)
    optimizer, scheduler, t_total = create_optimizer_and_scheduler(
        args,
        model,
        train_dataloader,
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size = %d", args.train_batch_size)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    total_train_loss = 0.0

    # Track interval-level statistics for periodic logging
    interval_loss = 0.0
    interval_steps = 0
    interval_train_preds = []
    interval_train_labels = []
    history_path = os.path.join(args.output_dir, "training_history.csv")

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc=f"Iteration-Epoch-{epoch + 1}")

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = prepare_batch_inputs(args, batch)

            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]

            # Track raw loss before gradient accumulation scaling
            raw_loss_value = loss.item()
            total_train_loss += raw_loss_value
            interval_loss += raw_loss_value
            interval_steps += 1

            # Store predictions for interval metrics
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
            interval_train_preds.append(probs)
            interval_train_labels.append(labels)

            # Scale loss when using gradient accumulation
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            should_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                len(train_dataloader) <= args.gradient_accumulation_steps
                and (step + 1) == len(train_dataloader)
            )

            if not should_step:
                continue

            # Update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # Periodic evaluation and history logging
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                evaluate_and_record_history(
                    args=args,
                    model=model,
                    dev_dataset=dev_dataset,
                    test_dataset=test_dataset,
                    global_step=global_step,
                    epoch=epoch,
                    interval_loss=interval_loss,
                    interval_steps=interval_steps,
                    interval_train_preds=interval_train_preds,
                    interval_train_labels=interval_train_labels,
                    scheduler=scheduler,
                    history_path=history_path,
                    leftover=False,
                )
                interval_loss = 0.0
                interval_steps = 0
                interval_train_preds = []
                interval_train_labels = []

            # Periodic checkpoint saving
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(
                    args,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    global_step,
                )

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    # Save any leftover interval that did not reach the next logging step
    if interval_steps > 0 and interval_train_preds and interval_train_labels:
        evaluate_and_record_history(
            args=args,
            model=model,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            global_step=global_step,
            epoch=epoch,
            interval_loss=interval_loss,
            interval_steps=interval_steps,
            interval_train_preds=interval_train_preds,
            interval_train_labels=interval_train_labels,
            scheduler=scheduler,
            history_path=history_path,
            leftover=True,
        )

    logger.info("Training history saved incrementally to %s", history_path)
    avg_train_loss = total_train_loss / max(global_step, 1)
    return global_step, avg_train_loss


def main(cli_args):
    """Load config, datasets, model, then run training/evaluation."""
    config_filename = f"{cli_args.taxonomy}.json"
    config_path = os.path.join(PROJECT_ROOT, "config", config_filename)

    with open(config_path, encoding="utf-8") as f:
        args = AttrDict(json.load(f))

    # Default revision if not explicitly provided in config
    if not hasattr(args, "hf_revision"):
        args.hf_revision = "main"

    init_logger()
    logger.info("Training/evaluation parameters %s", args)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args)

    # Load task labels
    processor = GoEmotionsProcessor(args)
    label_list = processor.get_labels()

    # Load tokenizer + model
    tokenizer, model, model_class = load_model_components(args, label_list)

    # Select device
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Load datasets if configured
    train_dataset = (
        load_and_cache_examples(args, tokenizer, mode="train")
        if args.train_file
        else None
    )
    dev_dataset = (
        load_and_cache_examples(args, tokenizer, mode="dev")
        if args.dev_file
        else None
    )
    test_dataset = (
        load_and_cache_examples(args, tokenizer, mode="test")
        if args.test_file
        else None
    )

    # Fall back to test evaluation if no dev set is available
    if dev_dataset is None:
        args.evaluate_test_during_training = True

    if args.do_train and train_dataset is not None:
        global_step, tr_loss = train(
            args,
            model,
            tokenizer,
            train_dataset,
            dev_dataset,
            test_dataset,
        )
        logger.info("global_step = %s, average train loss = %s", global_step, tr_loss)

    results = {}
    if args.do_eval and test_dataset is not None:
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(
                glob.glob(
                    os.path.join(args.output_dir, "**", "pytorch_model.bin"),
                    recursive=True,
                )
            )
        )

        if not getattr(args, "eval_all_checkpoints", False):
            checkpoints = checkpoints[-1:]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            step_id = checkpoint.split("-")[-1] if "-" in checkpoint else None

            checkpoint_kwargs = get_hf_load_kwargs(checkpoint, args.hf_revision)
            model = model_class.from_pretrained(checkpoint, **checkpoint_kwargs)
            model.to(args.device)

            result = evaluate(
                args,
                model,
                test_dataset,
                mode="test",
                global_step=step_id,
            )
            result = {
                k + (f"_{step_id}" if step_id else ""): v for k, v in result.items()
            }
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding="utf-8") as f_w:
            for key in sorted(results.keys()):
                f_w.write(f"{key} = {results[key]}\n")

        logger.info("Evaluation results saved to %s", output_eval_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taxonomy",
        type=str,
        required=True,
        help="Config name without .json",
    )
    cli_args = parser.parse_args()
    main(cli_args)