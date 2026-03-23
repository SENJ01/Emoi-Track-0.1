import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict

# transformers imports
from transformers import (
    BertConfig, BertTokenizer,
    DistilBertConfig, DistilBertTokenizer,
    RobertaConfig, RobertaTokenizer,
    get_linear_schedule_with_warmup
)

# Optimizer
from torch.optim import AdamW

# Model imports
from model import (
    BertForMultiLabelClassification,
    DistilBertForMultiLabelClassification,
    RobertaForMultiLabelClassification
)

from utils import (
    init_logger,
    set_seed,
    compute_metrics
)
from data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)

logger = logging.getLogger(__name__)

# ------------------ TRAIN FUNCTION ------------------ #
def train(args, model, tokenizer, train_dataset, dev_dataset=None, test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total
    )

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type.lower() == "bert":
                inputs["token_type_ids"] = batch[2]

            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

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
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step

# ------------------ EVALUATION FUNCTION ------------------ #
def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info(f"***** Running evaluation on {mode} dataset {'(' + str(global_step) + ' step)' if global_step else ''} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Eval Batch size = {args.eval_batch_size}")

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type.lower() == "bert":
                inputs["token_type_ids"] = batch[2]

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {"loss": eval_loss}
    preds[preds > args.threshold] = 1
    preds[preds <= args.threshold] = 0
    results.update(compute_metrics(out_label_ids, preds))

    # Save per-mode evaluation results
    mode_dir = os.path.join(args.output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    output_eval_file = os.path.join(mode_dir, f"{mode}-{global_step}.txt" if global_step else f"{mode}.txt")

    with open(output_eval_file, "w") as f_w:
        logger.info(f"***** Eval results on {mode} dataset *****")
        for key in sorted(results.keys()):
            logger.info(f"  {key} = {results[key]}")
            f_w.write(f"{key} = {results[key]}\n")

    return results

# ------------------ MAIN FUNCTION ------------------ #
def main(cli_args):
    # Load config
    config_filename = f"{cli_args.taxonomy}.json"
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    logger.info(f"Training/evaluation parameters {args}")

    # Ensure main output folder exists
    os.makedirs(args.output_dir, exist_ok=True)

    init_logger()
    set_seed(args)

    # Processor & labels
    processor = GoEmotionsProcessor(args)
    label_list = processor.get_labels()

    # ------------------ CONFIG & TOKENIZER ------------------ #
    if args.model_type.lower() == "bert":
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            finetuning_task=args.task,
            id2label={str(i): l for i, l in enumerate(label_list)},
            label2id={l: i for i, l in enumerate(label_list)}
        )
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model_class = BertForMultiLabelClassification
    elif args.model_type.lower() == "distilbert":
        config = DistilBertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            id2label={str(i): l for i, l in enumerate(label_list)},
            label2id={l: i for i, l in enumerate(label_list)}
        )
        tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model_class = DistilBertForMultiLabelClassification
    elif args.model_type.lower() == "roberta":
        config = RobertaConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(label_list),
            id2label={str(i): l for i, l in enumerate(label_list)},
            label2id={l: i for i, l in enumerate(label_list)}
        )
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name_or_path)
        model_class = RobertaForMultiLabelClassification
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Load model
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # ------------------ DATASET ------------------ #
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

    if dev_dataset is None:
        args.evaluate_test_during_training = True

    # ------------------ TRAINING ------------------ #
    if args.do_train and train_dataset:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset, dev_dataset, test_dataset)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    # ------------------ EVALUATION ------------------ #
    results = {}
    if args.do_eval and test_dataset:
        # Find checkpoints safely
        checkpoints = list(
            os.path.dirname(c) for c in sorted(
                glob.glob(os.path.join(args.output_dir, "**", "pytorch_model.bin"), recursive=True)
            )
        )
        if not getattr(args, "eval_all_checkpoints", False):
            checkpoints = checkpoints[-1:]

        logger.info(f"Evaluate the following checkpoints: {checkpoints}")

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if "-" in checkpoint else None
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = {k + (f"_{global_step}" if global_step else ""): v for k, v in result.items()}
            results.update(result)

        # Save combined eval results
        os.makedirs(args.output_dir, exist_ok=True)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write(f"{key} = {results[key]}\n")
        logger.info(f"Evaluation results saved to {output_eval_file}")

# ------------------ CLI ------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taxonomy", type=str, required=True, help="Config name without .json"
    )
    cli_args = parser.parse_args()
    main(cli_args)