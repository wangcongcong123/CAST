import os
import sys
from copy import deepcopy
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers.trainer_utils import EvaluationStrategy
from datasets import load_dataset, load_from_disk, load_metric
import torch, json
from tqdm import tqdm
from ptt import save_and_check_if_early_stop
from ptt import HFTrainer, check_output_path, add_filehandler_for_logger, HFTrainingArguments, get_existing_cks, select_dataset_dict, print_few_examples
from sklearn.metrics import classification_report

transformers.logging.set_verbosity_info()
logger = transformers.logging.get_logger()


def convert_to_features(example_batch, args, tokenizer):
    baseline_targets = []
    for text, event_name in zip(example_batch["text"], example_batch["event_name"]):
        if any(i in text.lower() for i in event_name.lower().split()):
            baseline_targets.append("yes")
        else:
            baseline_targets.append("no")

    encoded_source = tokenizer(example_batch["source"], padding=True, truncation=True, max_length=args.max_src_length)
    encoded_target = tokenizer(example_batch["target"], padding=True, truncation=True, max_length=args.max_tgt_length)
    source_lengths = [len(encoded_source["input_ids"][0])] * len(encoded_source["input_ids"])
    target_lengths = [len(encoded_target["input_ids"][0])] * len(encoded_target["input_ids"])

    encoded_source.update({"labels": encoded_target["input_ids"], "decoder_attention_mask": encoded_target["attention_mask"], "source_lengths": source_lengths, "target_lengths": target_lengths})
    encoded_source.update({"baseline_targets": baseline_targets, "targets": example_batch["target"]})
    return encoded_source


def collate_fn(examples):
    source_inputs = [{"input_ids": each["input_ids"], "attention_mask": each["attention_mask"]} for each in
                     examples]
    target_inputs = [{"input_ids": each["labels"], "attention_mask": each["decoder_attention_mask"]} for each in
                     examples]
    source_inputs_padded = tokenizer.pad(source_inputs, return_tensors='pt')
    target_inputs_padded = tokenizer.pad(target_inputs, return_tensors='pt')
    source_inputs_padded.update({"labels": target_inputs_padded["input_ids"],
                                 "decoder_attention_mask": target_inputs_padded["attention_mask"]})
    return source_inputs_padded


def evaluate(args, logger, model, tokenizer, eval_dataloader, steps=0, tag="epoch", is_test=False):
    print_few_examples(logger, eval_dataloader.dataset, n=3, tokenizer=tokenizer)
    model.eval().to(args.device)
    gts = []
    preds = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="evaluating..."):
        with torch.no_grad():
            batch.to(args.device)
            if isinstance(model, torch.nn.DataParallel):
                predictions = model.module.generate(input_ids=batch["input_ids"], max_length=args.max_tgt_length,
                                                    attention_mask=batch["attention_mask"])
            else:
                predictions = model.generate(input_ids=batch["input_ids"], max_length=args.max_tgt_length,
                                             attention_mask=batch["attention_mask"])
            pred = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predictions]
            gt = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["labels"]]
            preds.extend(pred)
            gts.extend(gt)

    logger.info(f":{classification_report(gts, preds, digits=4)}")
    metrics_fn = load_metric(f"{args.relative_script_dir}/metric_scripts/cls.py", "long")
    metrics = metrics_fn.compute(predictions=preds, references=gts)
    logger.info(f"val_cls_report:\n {json.dumps(metrics, indent=2)}")
    eval_score = metrics[args.eval_on]
    logger.info(f"val_{args.eval_on}_score: {eval_score}")
    is_early_stop = False
    if not is_test:
        is_early_stop = save_and_check_if_early_stop(eval_score, args, logger, model, tokenizer, steps=steps, tag=tag)
    return {"eval_scores": metrics, "preds": preds,
            "is_early_stop": is_early_stop}


if __name__ == '__main__':
    args = HFTrainingArguments(
        output_path='outputs',
        # we tried larger epoch number like 16 that gave no further improvement
        num_train_epochs=6,
        save_total_limit=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps_or_ratio=0.1,
        weight_decay=0.01,
        log_and_save_steps=100,
        fp16=True,
        learning_rate=5e-05,
        gradient_accumulation_steps=1,
        visible_gpu_devices="0",
        scheduler="warmuplinear",
        evaluation_strategy=EvaluationStrategy.EPOCH,
        logging_dir="runs",
        do_train=True,
        do_test=True,
        # when do_test = True, eval on best based on validation during training
        ck_index_select=0,
    )
    # this code presents an example of using training set of "queensland_floods" as the training source event data for fine-tuning t5-small and test it on the test of the same event (in-domain)
    # to test it on "nepal_earthquake", just change dataset_name to "nepal_earthquake" and set args.do_train to be False.
    args.max_src_length = 128
    args.max_tgt_length = 10
    # 't5-small' or 't5-base'
    args.model_select = 't5-small'
    dataset_name = "queensland_floods"
    # 't2t' or 'normal'
    data_config = "t2t"

    args.dataset = (f'{args.relative_script_dir}/data_scripts/{dataset_name}.py', data_config)
    args.output_path = os.path.join(args.output_path, f"{args.model_select}_{dataset_name}_{data_config}")

    if args.do_train:
        check_output_path(args.output_path, force=False)
        add_filehandler_for_logger(args.output_path, logger)
    else:
        add_filehandler_for_logger(args.output_path, logger, out_name="test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_select)
    try:
        dataset = load_dataset(*args.dataset)
    except:
        dataset = load_dataset(*args.dataset)

    logger.info(f"load model from {args.model_select}")
    model = AutoModelWithLMHead.from_pretrained(args.model_select)

    cache_foldername = args.output_path.split(os.sep)[-1]
    cache_path = os.path.join(args.working_dir, ".cache", cache_foldername)
    if not os.path.isdir(cache_path):
        fn_kwargs = {"args": args, "tokenizer": tokenizer}
        dataset = select_dataset_dict(dataset, args)
        encoded_dataset = dataset.map(convert_to_features, batched=True, fn_kwargs=fn_kwargs, num_proc=6)
        encoded_dataset.save_to_disk(cache_path)
    else:
        encoded_dataset = load_from_disk(cache_path)

    encoded = deepcopy(encoded_dataset)
    columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
    encoded.set_format(type='torch', columns=columns)
    train_dataset = encoded["train"] if "train" in encoded else None
    test_dataset = encoded["validation"] if "validation" in encoded else None

    if args.do_train:
        trainer = HFTrainer(
            logger=logger,
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=collate_fn,
            evaluate_fn=evaluate,
            tokenizer=tokenizer
        )
        trainer.train(verbose=True)

    if args.do_test:
        sorted_indices, index2path = get_existing_cks(args.output_path, return_best_ck=False)
        if args.ck_index_select < 0:
            model_path = index2path[sorted_indices[args.ck_index_select]]
        else:
            bests = [name for name in os.listdir(args.output_path) if name.startswith("best")]
            if bests != []:
                model_path = os.path.join(args.output_path, bests[0])
            else:
                model_path = index2path[sorted_indices[args.ck_index_select]]

        model = AutoModelWithLMHead.from_pretrained(model_path)
        logger.info("-------------------eval and predict on test set-------------------")
        test_dataloader = torch.utils.data.DataLoader(encoded["test"], collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size * args.n_gpu, shuffle=False)
        eval_dict = evaluate(args, logger, model, tokenizer, test_dataloader, is_test=True)
