import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import logging
import glob

import numpy as np
import pandas as pd
import torch
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

from torch.optim import AdamW

from models.model import (
    BertForMultiLabelClassification,
    DistilBertForMultiLabelClassification,
    RobertaForMultiLabelClassification,
)

from utils.utils import init_logger, set_seed, compute_metrics
from data.data_loader import load_and_cache_examples, GoEmotionsProcessor

logger = logging.getLogger(__name__)


def evaluate(args, model, eval_dataset, mode, global_step=None, save_results=True):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    logger.info(
        f"***** Running evaluation on {mode} dataset "
        f"{'(' + str(global_step) + ' step)' if global_step else ''} *****"
    )
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Eval Batch size = {args.eval_batch_size}")

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc=f"Evaluating-{mode}"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
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

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = inputs["labels"].detach().cpu().numpy()

        if preds is None:
            preds = probs
            out_label_ids = labels
        else:
            preds = np.append(preds, probs, axis=0)
            out_label_ids = np.append(out_label_ids, labels, axis=0)

    eval_loss = eval_loss / max(nb_eval_steps, 1)
    hard_preds = (preds > args.threshold).astype(int)

    results = {"loss": eval_loss}
    results.update(compute_metrics(out_label_ids, hard_preds))

    if save_results:
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        output_eval_file = os.path.join(
            mode_dir, f"{mode}-{global_step}.txt" if global_step else f"{mode}.txt"
        )

        with open(output_eval_file, "w", encoding="utf-8") as f_w:
            logger.info(f"***** Eval results on {mode} dataset *****")
            for key in sorted(results.keys()):
                logger.info(f"  {key} = {results[key]}")
                f_w.write(f"{key} = {results[key]}\n")

    return results


def append_training_history_row(history_path, row):
    row_df = pd.DataFrame([row])
    row_df.to_csv(
        history_path,
        mode="a",
        header=not os.path.exists(history_path),
        index=False,
    )


def train(args, model, tokenizer, train_dataset, dev_dataset=None, test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

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
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total,
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

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type.lower() == "bert":
                inputs["token_type_ids"] = batch[2]

            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]

            raw_loss_value = loss.item()
            total_train_loss += raw_loss_value
            interval_loss += raw_loss_value
            interval_steps += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()

            interval_train_preds.append(probs)
            interval_train_labels.append(labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                len(train_dataloader) <= args.gradient_accumulation_steps
                and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    train_preds = np.vstack(interval_train_preds)
                    train_labels = np.vstack(interval_train_labels)
                    train_preds_bin = (train_preds > args.threshold).astype(int)
                    train_metrics = compute_metrics(train_labels, train_preds_bin)

                    avg_train_loss = interval_loss / max(interval_steps, 1)

                    eval_dataset = (
                        test_dataset
                        if args.evaluate_test_during_training
                        else dev_dataset
                    )
                    eval_mode = "test" if args.evaluate_test_during_training else "dev"

                    eval_results = evaluate(
                        args,
                        model,
                        eval_dataset,
                        eval_mode,
                        global_step=global_step,
                        save_results=True,
                    )

                    row = {
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

                    append_training_history_row(history_path, row)
                    logger.info(
                        f"Appended training history at step {global_step} to {history_path}"
                    )

                    interval_loss = 0.0
                    interval_steps = 0
                    interval_train_preds = []
                    interval_train_labels = []

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)

                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))

                    if args.save_optimizer:
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(ckpt_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(ckpt_dir, "scheduler.pt"),
                        )

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    # Save final leftover interval if training ended before the next logging step
    if interval_steps > 0 and len(interval_train_preds) > 0 and len(interval_train_labels) > 0:
        train_preds = np.vstack(interval_train_preds)
        train_labels = np.vstack(interval_train_labels)
        train_preds_bin = (train_preds > args.threshold).astype(int)
        train_metrics = compute_metrics(train_labels, train_preds_bin)

        avg_train_loss = interval_loss / max(interval_steps, 1)

        eval_dataset = test_dataset if args.evaluate_test_during_training else dev_dataset
        eval_mode = "test" if args.evaluate_test_during_training else "dev"

        eval_results = evaluate(
            args,
            model,
            eval_dataset,
            eval_mode,
            global_step=global_step,
            save_results=True,
        )

        row = {
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

        append_training_history_row(history_path, row)
        logger.info(
            f"Appended final leftover training history at step {global_step} to {history_path}"
        )

    logger.info(f"Training history saved incrementally to {history_path}")

    avg_train_loss = total_train_loss / max(global_step, 1)
    return global_step, avg_train_loss


def main(cli_args):
    config_filename = f"{cli_args.taxonomy}.json"
    config_path = os.path.join(PROJECT_ROOT, "config", config_filename)

    with open(config_path, encoding="utf-8") as f:
        args = AttrDict(json.load(f))

    init_logger()
    logger.info(f"Training/evaluation parameters {args}")

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args)

    processor = GoEmotionsProcessor(args)
    label_list = processor.get_labels()

    if args.model_type.lower() == "bert":
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): l for i, l in enumerate(label_list)},
            label2id={l: i for i, l in enumerate(label_list)},
        )
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model_class = BertForMultiLabelClassification

    elif args.model_type.lower() == "distilbert":
        config = DistilBertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            id2label={str(i): l for i, l in enumerate(label_list)},
            label2id={l: i for i, l in enumerate(label_list)},
        )
        tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model_class = DistilBertForMultiLabelClassification

    elif args.model_type.lower() == "roberta":
        config = RobertaConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            id2label={str(i): l for i, l in enumerate(label_list)},
            label2id={l: i for i, l in enumerate(label_list)},
        )
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model_class = RobertaForMultiLabelClassification

    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

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

    if dev_dataset is None:
        args.evaluate_test_during_training = True

    if args.do_train and train_dataset is not None:
        global_step, tr_loss = train(
            args, model, tokenizer, train_dataset, dev_dataset, test_dataset
        )
        logger.info(f"global_step = {global_step}, average train loss = {tr_loss}")

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

        logger.info(f"Evaluate the following checkpoints: {checkpoints}")

        for checkpoint in checkpoints:
            step_id = checkpoint.split("-")[-1] if "-" in checkpoint else None
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            result = evaluate(
                args, model, test_dataset, mode="test", global_step=step_id
            )
            result = {
                k + (f"_{step_id}" if step_id else ""): v
                for k, v in result.items()
            }
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w", encoding="utf-8") as f_w:
            for key in sorted(results.keys()):
                f_w.write(f"{key} = {results[key]}\n")

        logger.info(f"Evaluation results saved to {output_eval_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taxonomy", type=str, required=True, help="Config name without .json"
    )
    cli_args = parser.parse_args()
    main(cli_args)