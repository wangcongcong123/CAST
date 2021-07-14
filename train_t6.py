import os
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
    encoded_source = tokenizer(example_batch["source"], padding=True, truncation=True,
                               max_length=args.max_src_length)
    encoded_target = tokenizer(example_batch["target"], padding=True, truncation=True,
                               max_length=args.max_tgt_length)

    source_lengths = [len(encoded_source["input_ids"][0])] * len(encoded_source["input_ids"])
    target_lengths = [len(encoded_target["input_ids"][0])] * len(encoded_target["input_ids"])

    encoded_source.update(
        {"labels": encoded_target["input_ids"], "decoder_attention_mask": encoded_target["attention_mask"], "source_lengths": source_lengths, "target_lengths": target_lengths})
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


def evaluate(args, logger, model, tokenizer, eval_dataloader, steps=0, tag="epoch", is_test=False, verbose=True):
    if verbose:
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
    logger.info(f"val_cls_report: {json.dumps(metrics, indent=2)}")
    eval_score = metrics[args.eval_on]
    logger.info(f"val_{args.eval_on}_score: {eval_score}")
    is_early_stop = False
    if not is_test:
        is_early_stop = save_and_check_if_early_stop(eval_score, args, logger, model, tokenizer, steps=steps, tag=tag)
    return {"eval_scores": metrics, "preds": preds,
            "is_early_stop": is_early_stop}


if __name__ == '__main__':
    overall_report = {}
    EVENTSHORT2LONG = {"QF": "Queensland Floods", "SH": "Sandy Hurricane", "AF": "Alberta Floods", "BB": "Boston Bombings", "OT": "Oklahoma Tornado", "WTE": "West Texas Explosion"}
    # train_event_names => source events
    # below is an example of leave-one-out crisis domain adaptation, feel free to change it for your needs
    train_event_names = ["AF-BB-WTE-QF-OT", "SH-BB-WTE-QF-OT", "SH-AF-WTE-QF-OT", "SH-AF-BB-QF-OT", "SH-AF-BB-WTE-OT", "SH-AF-BB-WTE-QF"]
    # train_event_names = ["AF-SH-OT","BB-WTE"]
    # train_event_names = ["QF-SH-OT"]
    # train_event_names = ["BB-SH-OT"]

    for train_event_name in train_event_names:
        # test_event_name => target event
        # below is an example of leave-one-out crisis domain adaptation, feel free to configure it for your needs
        t_ens = list(EVENTSHORT2LONG.keys())
        for i in train_event_name.split("-"):
            t_ens.remove(i)
        test_event_name = "-".join(t_ens)
        # test_event_name = "WTE"
        args = HFTrainingArguments(
            output_path='outputs',
            num_train_epochs=12,
            save_total_limit=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps_or_ratio=0.1,
            weight_decay=0.01,
            log_and_save_steps=200,
            fp16=True,
            learning_rate=5e-05,
            gradient_accumulation_steps=1,
            visible_gpu_devices="0",
            scheduler="warmuplinear",
            # evaluation_strategy=EvaluationStrategy.NO,
            evaluation_strategy=EvaluationStrategy.EPOCH,
            logging_dir=None,
            do_train=True,
            do_test=True,
        )
        # when do_test=True, select the best ck
        args.ck_index_select = 0
        args.max_src_length = 128
        args.max_tgt_length = 10
        # 't5-small' or 't5-base'
        args.model_select = 't5-small'
        dataset_name = "crisis_t6"
        # 't2t' (postQ) or 'normal' (standard)
        data_config = "t2t"

        train_event_list = train_event_name.split("-")
        test_event_list = test_event_name.split("-")

        assert all(e in EVENTSHORT2LONG for e in train_event_list)
        assert all(e in EVENTSHORT2LONG for e in test_event_list)
        if set(train_event_list) != set(test_event_list):
            assert set(train_event_list).intersection(set(test_event_list)) == set()

        args.dataset = (f'{args.relative_script_dir}/data_scripts/{dataset_name}.py', data_config)
        args.output_path = os.path.join(args.output_path, f"{args.model_select}_{dataset_name}_{data_config}_{train_event_name}{'_self' if train_event_name == test_event_name else ''}")

        if args.do_train:
            check_output_path(args.output_path, force=False)
            add_filehandler_for_logger(args.output_path, logger)
        else:
            add_filehandler_for_logger(args.output_path, logger, out_name="test")

        tokenizer = AutoTokenizer.from_pretrained(args.model_select)
        try:
            dataset = load_dataset(*args.dataset)  # inconsistent if use data_files
        except:
            dataset = load_dataset(*args.dataset)

        model = AutoModelWithLMHead.from_pretrained(args.model_select)
        cache_foldername = args.output_path.split(os.sep)[-1].rstrip(f"{train_event_name}_{test_event_name}")
        cache_path = os.path.join(args.working_dir, ".cache", cache_foldername)

        if not os.path.isdir(cache_path):
            fn_kwargs = {"args": args, "tokenizer": tokenizer}
            dataset = select_dataset_dict(dataset, args)
            encoded_dataset = dataset.map(convert_to_features, batched=True, fn_kwargs=fn_kwargs, num_proc=6)
            encoded_dataset.save_to_disk(cache_path)
        else:
            encoded_dataset = load_from_disk(cache_path)


        def run_train(encoded, model):
            columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask', 'target_lengths', 'source_lengths']
            encoded.set_format(type='torch', columns=columns)
            train_dataset = encoded["train"] if "train" in encoded else None
            logger.info(f"actual train max source length: {max(train_dataset['source_lengths'])}")
            logger.info(f"actual train max target length: {max(train_dataset['target_lengths'])}")
            eval_dataset = encoded["validation"] if "validation" in encoded else None
            trainer = HFTrainer(
                logger=logger,
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collate_fn,
                evaluate_fn=evaluate,
                tokenizer=tokenizer
            )
            trainer.train(verbose=True)


        def run_test(encoded, tag=""):
            sorted_indices, index2path = get_existing_cks(args.output_path, return_best_ck=False)
            if args.ck_index_select < 0:
                model_path = index2path[sorted_indices[args.ck_index_select]]
            else:
                bests = [name for name in os.listdir(args.output_path) if name.startswith("best")]
                if bests != []:
                    model_path = os.path.join(args.output_path, bests[0])
                else:
                    try:
                        model_path = index2path[sorted_indices[args.ck_index_select]]
                    except:
                        raise ValueError("No checkpoints found, you may have not trained the model yet.")

            model = AutoModelWithLMHead.from_pretrained(model_path)
            # model = AutoModelWithLMHead.from_pretrained(args.model_select)
            # let's eval the end of training weights: good for development stage
            logger.info(f"-------------------eval and predict on test set of ({train_event_name} => {test_event_name} {tag})-------------------")
            test_dataloader = torch.utils.data.DataLoader(encoded["test"], collate_fn=collate_fn,
                                                          batch_size=args.per_device_eval_batch_size * args.n_gpu,
                                                          shuffle=False)
            eval_dict = evaluate(args, logger, model, tokenizer, test_dataloader, is_test=True, verbose=False)  # does not help: dataset["test"]["ex_id"]
            return eval_dict


        encoded = deepcopy(encoded_dataset)
        eval_scores_summary = {}
        if train_event_name == test_event_name:
            # for in-domain adaptation, we use 5-fold validation
            # in this case, we make sure train and test
            assert args.do_train and args.do_test
            # five fold validation if train_event_name == test_event_name, namely, evaluate on itself
            from sklearn.model_selection import KFold

            encoded_dataset["train"] = encoded_dataset["train"].filter(lambda example: any(example["event_name"] == EVENTSHORT2LONG[e] for e in train_event_list)).shuffle(2020)
            indices = range(len(encoded_dataset["train"]))
            kfold = KFold(n_splits=5)
            fold_index = 1
            eval_scores_summary = {}
            for train_indices, test_indices in kfold.split(indices):
                logger.info(f"*************5-fold training on index = {fold_index}**********")
                # we have to re-initialize model here for 5-fold validation
                model = AutoModelWithLMHead.from_pretrained(args.model_select)
                encoded["train"] = encoded_dataset["train"].select(train_indices)
                encoded["test"] = encoded_dataset["train"].select(test_indices)
                run_tag = f"5-fold-{fold_index}"

                run_train(encoded, model)
                eval_dict = run_test(encoded, tag=run_tag)

                eval_scores_summary[run_tag] = eval_dict["eval_scores"]
                fold_index += 1
            logger.info(f"*************5-fold eval_scores_summary using dataset {train_event_name} with data_config {data_config}**********")
            logger.info(f"{json.dumps(eval_scores_summary, indent=2)}")
            overall_report[f"train_on_{train_event_name}"] = eval_scores_summary
        else:
            # for cross domain adaptation, we use the source event(s) as the training data and test it directly on the target event
            # filter by event names to select train and test set when they are not equal, train using the dataset specified by train_event_name and evaluation on the dataset specified by test_event_name
            encoded["train"] = encoded_dataset["train"].filter(lambda example: any(example["event_name"] == EVENTSHORT2LONG[e] for e in train_event_list))
            if args.do_train:
                run_train(encoded, model)
            if args.do_test:
                for test_e in test_event_list:
                    encoded["test"] = encoded_dataset["train"].filter(lambda example: example["event_name"] == EVENTSHORT2LONG[test_e])
                    run_tag = f"{train_event_name}->{test_e}"
                    eval_dict = run_test(encoded, tag=run_tag)
                    eval_scores_summary[run_tag] = eval_dict["eval_scores"]
                logger.info(
                    f"*************eval_scores_summary using training dataset {train_event_name} and tested on {test_event_name} with data_config {data_config}**********")
                logger.info(f"{json.dumps(eval_scores_summary, indent=2)}")
                overall_report[f"train_on_{train_event_name}"] = eval_scores_summary

        logger.info(f"*************so-far overall_report with data_config {data_config}**********")
        logger.info(f"{json.dumps(overall_report, indent=2)}")
