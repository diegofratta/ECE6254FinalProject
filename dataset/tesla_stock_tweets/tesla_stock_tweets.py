# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Twitter Tesla Stock Dataset."""

import datasets
from datasets.tasks import TextClassification
import csv

_DESCRIPTION = """\
Twitter Stock Dataset.
"""

_CITATION = """\
@InProceedings{todo:todo,
title = {todo},
author={todo},
year={2022}
}
"""

_DESCRIPTION = """\
TODO: Add description of the dataset here
"""

_HOMEPAGE = "TODO: Add a link to an official homepage for the dataset here"
_LICENSE = "TODO: Add a link to an official homepage for the dataset here"

_DOWNLOAD_URL = "https://luisp23.github.io/assets/ece-6254-project/"
_DOWNLOAD_URLS = {
    "train": _DOWNLOAD_URL + "tesla_stock_tweets_train.csv",
    "test": _DOWNLOAD_URL + "tesla_stock_tweets_test.csv",
    "val": _DOWNLOAD_URL + "tesla_stock_tweets_val.csv",
}


class TeslaStockTweetsConfig(datasets.BuilderConfig):
    """BuilderConfig for TeslaStockTweets."""

    def __init__(self, **kwargs):
        """BuilderConfig for StockTweets.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TeslaStockTweetsConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class TeslaStockTweets(datasets.GeneratorBasedBuilder):
    """Stock tweets dataset."""

    BUILDER_CONFIGS = [
        TeslaStockTweetsConfig(name="plain_text", description="Plain text",)
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {"text": datasets.Value("string"), "label": datasets.features.ClassLabel(names=["neg", "neut", "pos"])}
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[TextClassification(text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_DOWNLOAD_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": downloaded_files["test"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": downloaded_files["val"]}),
        ]

    def _generate_examples(self, filepath):
        """Generate StockTweets examples."""
        print("generating examples from = %s", filepath)
        with open(filepath, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                line_count +=1

                yield line_count, {"text": row["Tweet"], "label": int(row["Sentiment"])}
