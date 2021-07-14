# coding=utf-8
# This script is finished following HF's datasets' template:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
# More examples as references to write a customized dataset can be found here:
# https://github.com/huggingface/datasets/tree/master/datasets

from __future__ import absolute_import, division, print_function
import json
import os

import datasets

_CITATION = """\

"""
_DESCRIPTION = """\
"""

from ptt.utils import LOCAL_DATA_LOAD_DIR

_TRAIN_DOWNLOAD_URL = f"{LOCAL_DATA_LOAD_DIR}/data/nepal_earthquake/2015_Nepal_Earthquake_train_new.json"
_VAL_DOWNLOAD_URL = f"{LOCAL_DATA_LOAD_DIR}/data/nepal_earthquake/2015_Nepal_Earthquake_dev_new.json"
_TEST_DOWNLOAD_URL = f"{LOCAL_DATA_LOAD_DIR}/data/nepal_earthquake/2015_Nepal_Earthquake_test_new.json"

NORMAL_LABELS_MAP = {"relevant": 'relevant', "not_relevant": 'not relevant'}
LABELS_MAP = {"relevant": 'yes', "not_relevant": 'no'}

class NepalQueenslandConfig(datasets.BuilderConfig):
    def __init__(
            self,
            **kwargs,
    ):
        super(NepalQueenslandConfig, self).__init__(version=datasets.Version("0.0.0", ""), **kwargs)

class NepalQueensland(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        NepalQueenslandConfig(
            name="normal",
            description="text as the source and sentiment label as the target",
        ),
        NepalQueenslandConfig(
            name="t2t",
            description="target is the sentiment label and source is constructed by a multi-choice QA template: context: ... question: ... choices: ",
        ),
    ]

    """customize dataset."""
    # VERSION = datasets.Version("0.0.0")
    def _info(self):
        data_info = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "event_name": datasets.Value("string"),
                    "set_name": datasets.Value("string"),
                    "question_template": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "source": datasets.Value("string"),
                    "target": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="#",
            citation=_CITATION,
        )
        # data_info.__class__.question = QUESTION
        return data_info

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
            val_path = dl_manager.download_and_extract(_VAL_DOWNLOAD_URL)
            test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)

            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": val_path}),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),

            ]
        else:
            splits = []
            train_path = dl_manager.download_and_extract(self.config.data_files["train"])
            splits.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}))
            if "validation" in self.config.data_files:
                if os.path.isfile(self.config.data_files["validation"]):
                    val_path = dl_manager.download_and_extract(self.config.data_files["validation"])
                    splits.append(
                        datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": val_path}))
            if "test" in self.config.data_files:
                if os.path.isfile(self.config.data_files["test"]):
                    test_path = dl_manager.download_and_extract(self.config.data_files["test"])
                    splits.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}))
            return splits

    def _generate_examples(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "normal":
                    yield id_, {
                        "event_name": data["event_name"],
                        "set_name": data["set_name"],
                        "question_template": data["question_template"],
                        "question": data["question"],
                        "text": data["text"],
                        "source": data["text"],
                        "target": LABELS_MAP[data["label"]],
                    }
                else:
                    yield id_, {
                        "event_name": data["event_name"],
                        "set_name": data["set_name"],
                        "question_template": data["question_template"],
                        "question": data["question"],
                        "text": data["text"],
                        "source": data["source"],
                        "target": data["target"],
                    }
