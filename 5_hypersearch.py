from urllib.parse import urlparse
from collections import Counter
from itertools import product
from pprint import pprint
from typing import List  # python3.8
import random
import re

import os

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from loguru import logger
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import numpy as np
import torch
import wandb
import gc

from constants import (
    PERMITTED_DOMAINS,
    SOCIAL_MEDIA,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv(find_dotenv())


#Search text config defaults, only applies for 'processed_extra' datasets
SEARCH_MAX_SIZE = 5 # Max size of search results for each sample
SEARCH_ADD_LINK = True # Add the link to the search text
SEARCH_REDUCE_VERBOSITY = True # Verbosity of the search text, I recommend True for BERT and False for LLM's
SEARCH_FILTER_DOMAIN = False
SEARCH_FILTER_SOCIAL_MEDIA = False

VERSION = f"v6_{SEARCH_MAX_SIZE}_{SEARCH_FILTER_DOMAIN}_{SEARCH_FILTER_SOCIAL_MEDIA}"
DATA_DIR = "data/parquet/split"

filtered = Counter()

def prepare_data(
        config: dict,
        search_max_size: int = SEARCH_MAX_SIZE,
        search_add_link: bool = SEARCH_ADD_LINK,
        search_reduce_verbosity: bool = SEARCH_REDUCE_VERBOSITY,
        search_filter_domains: bool = SEARCH_FILTER_DOMAIN,
        search_filter_social_media: bool = SEARCH_FILTER_SOCIAL_MEDIA,
    ) -> List[pd.DataFrame]:

    if config["data"] == "processed" or config["data"] == "processed_extra":
        data_file_name = f"{config['dataset']}.parquet"
    elif config["data"] == "raw":
        data_file_name = f"{config['dataset']}_raw.parquet"
    else:
        raise ValueError("data can be only processed, processed_extra, raw")

    data_file_name = os.path.join(DATA_DIR, data_file_name)
    logger.debug(data_file_name)

    data = pd.read_parquet(data_file_name)

    if config["data"] != "processed_extra":
        data = data[["text_no_url", "label", "new_split"]]
        data.columns = ["text", "labels", "new_split"]
    else:
        cse_result = data["google_search_results"].apply(
            lambda rs: rs.tolist() if isinstance(rs, np.ndarray) else []
        )

        def clean_text(text:str) -> str:
            if text is None:
                return None
            
            text = re.sub(r"</?b>|&nbsp;|&#39|&quot|&middot", "", text)
            text = re.sub(r"\s\s+", " ", text)
            text = text.replace("... ...", "...")
            
            return text.strip()
        
        def convert_to_text(results: list) -> str:
            if search_filter_domains or search_filter_social_media:
                for r in results.copy():
                    r["link"] = urlparse(r["link"]).netloc
                    r["link"] = r["link"].replace("www.", "")
                    
                    if search_filter_social_media and any(d in r["link"] for d in SOCIAL_MEDIA):
                        results.remove(r)
                    elif search_filter_domains and not any(d in r["link"] for d in PERMITTED_DOMAINS):
                        filtered.update([r["link"]])
                        results.remove(r)

            if len(results) > search_max_size:
                results = results[:search_max_size]
                        
            
            if len(results) == 0:
                if not search_reduce_verbosity:
                    return "Sem resultados"
                else:
                    return ""
            
            r_texts = []
            for r in results:
                r_text = ""
                link = r.get("link")
                
                title = clean_text(r.get("title"))
                snippet = clean_text(r.get("snippet"))

                if not search_reduce_verbosity:
                    if title:
                        r_text += f"  Título: {title}\n"
                    if search_add_link and link:
                        r_text += f"  Link: {link}\n"
                    if snippet:
                        r_text += f"  Trechos: {snippet}"
                    r_texts.append(r_text.rstrip())
                else:
                    if title:
                        r_text += f"{title}\n"
                    if search_add_link and link:
                        r_text += f"{link}\n"
                    if snippet:
                        r_text += f"    {snippet}"
                    r_texts.append(r_text.rstrip())
            
            text = ""
            for i, r in enumerate(r_texts):
                if not search_reduce_verbosity:
                    text += f"- Resultado de busca {i+1}:\n{r}\n\n"
                else:
                    text += f"- Resultado {i+1}:\n{r}\n\n"
            
            return text.strip()
        
        data["cse_result_processed"] = cse_result.apply(convert_to_text)

        data = data[["text_no_url", "cse_result_processed", "label", "new_split"]]
        data.columns = ["text_a", "text_b", "labels", "new_split"]

    assert data["labels"].notna().all()

    train = data[data["new_split"] == "train"]
    train.name = "train"
    train.drop(columns=["new_split"], inplace=True)

    dev = data[data["new_split"] == "dev"]
    dev.name= "dev"
    dev.drop(columns=["new_split"], inplace=True)

    test = data[data["new_split"] == "test"]
    test.name = "test"
    test.drop(columns=["new_split"], inplace=True)

    return train, dev, test

def delete_checkpoints(output_dir: str):
    for root, _, files in os.walk(output_dir):   
        for filename in files:
            filepath = os.path.join(root, filename)
            mark_remove = True
            if os.path.isdir(filepath):
                mark_remove = False
            if filename in ['eval_results.txt', 'test.tsv', 'training_progress_scores.csv']:
                mark_remove = False
            if os.path.realpath(root) == os.path.realpath(output_dir) and filename in ['config.json', 'model_args.json', 'training_args.bin']:
                mark_remove = False

            if mark_remove:
                os.remove(filepath)


def train_model(config: dict):
    experiment_name = "_".join(str(p).replace('.', '-') for p in config.values())+'_' + VERSION
    output_dir = os.path.join("output", experiment_name)

    train, dev, test = prepare_data(config)

    logger.debug("Training set")
    logger.debug(train.head(3).to_dict("records"))

    logger.debug("Development set")
    logger.debug(dev.head(3).to_dict("records"))

    logger.debug("Testing set")
    logger.debug(test.head(3).to_dict("records"))

    logger.debug(output_dir)

    steps_per_epoch = int(len(train)/config['train_batch_size'])

    if os.path.exists(os.path.join(output_dir, "eval_results.txt")):
        return
    try:
        model_args = ClassificationArgs(
            # fixed params
            manual_seed=2025,
            max_seq_length=512,
            weight_decay=0.01,
            warmup_ratio=0.06,
            #warmup_steps=int(0.1*85*5), #42 steps
            #https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
            scheduler="linear_schedule_with_warmup",
            labels_list=["fake", "true"],
            fp16=True,

            # variable params
            eval_batch_size=config["train_batch_size"],
            train_batch_size=config["train_batch_size"],
            learning_rate=config["learning_rate"],
            num_train_epochs=10,
            #https://github.com/ThilinaRajapakse/simpletransformers/issues/1032
            #https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/configuration_bert.py#L79
            #https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/classification/transformer_models/bert_model.py#L40C42-L40C61
            config={
                "classifier_dropout": config["dropout"],
                "hidden_dropout_prob": config["dropout"],
                "attention_probs_dropout_prob": config["dropout"],
                "dropout": config["dropout"], #just for fallback
            },

            # logging
            wandb_project="false_br_search2",
            wandb_kwargs={
                "name": experiment_name,
                "tags": [config["data"], "text_no_url", config["dataset"], "full_dropout", VERSION],
            },
            logging_steps=10,

            # training evaluation
            evaluate_during_training=True,
            evaluate_during_training_steps=int(steps_per_epoch/2), #evaluate each half epoch
            evaluate_during_training_verbose=True,
            evaluate_each_epoch=False,

            # early stopping
            use_early_stopping=True,
            early_stopping_metric="f1_score",
            early_stopping_metric_minimize=False,
            early_stopping_patience=3,
            early_stopping_consider_epochs=False,
            early_stopping_delta=0.001,
            
            # no multiprocessings
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False,
            dataloader_num_workers=1,
            process_count=1,
            
            # saving
            overwrite_output_dir=True,
            output_dir=output_dir,
            best_model_dir=os.path.join(output_dir, "best_model"),
            #available_experiments=f"{output_dir}/best_model", ??
            no_save=False,
            save_steps=-1,
            save_model_every_epoch=False,
            save_best_model=True,
            save_eval_checkpoints=True,
            save_optimizer_and_scheduler=False,
        )

        model = ClassificationModel(
            "bert",
            "neuralmind/bert-base-portuguese-cased",
            args=model_args,
            use_cuda=True
        )

        logger.debug(model)
        logger.debug(model.model)

        model.train_model(train, eval_df=dev)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        logger.debug("Loading Best Model...")
        best_model = ClassificationModel(
            "bert",
            os.path.join(output_dir, "best_model"),
            args=model_args,
            use_cuda=True
        )

        logger.debug("Running evaluation on best model...")
        result, model_outputs, wrong_preds = best_model.eval_model(
            test, output_dir=output_dir
        )

        res_dict = {}
        for k,v in result.items():
            if isinstance(v, float) or isinstance(v, int):
                res_dict[f"test/{k}"] = v

        wandb.log(res_dict)

        test[["fake", "true"]] = model_outputs
        wrong_preds = {pred.guid for pred in wrong_preds}
        test["pred_correct"] = test.apply(
            lambda row: test.index.get_loc(row.name) not in wrong_preds, axis=True
        )
        test.to_csv(os.path.join(output_dir, "test.tsv"), sep="\t")

        #https://github.com/ThilinaRajapakse/simpletransformers/issues/1463
        wandb.finish()

        delete_checkpoints(output_dir)
        del model_outputs, wrong_preds, best_model

        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        delete_checkpoints(output_dir)
        raise e

if __name__ == "__main__":
    #lrs = [7.5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    lrs = [1e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4]
    batch_sizes = [8, 16]
    dropout = [0.1, 0.2, 0.3]

    params = {
        "data": ["processed_extra"],
        "dataset": ["COVID19.BR", "Fake.br"],
        "learning_rate": lrs,
        "train_batch_size": batch_sizes,
        "dropout":dropout
    }

    available_experiments = []
    for experiment in product(*params.values()):
        available_experiments.append(experiment)

    logger.debug(f"Grid-Search: {len(available_experiments)} runs")

    random.shuffle(available_experiments)
    for experiment in available_experiments:
        config = dict(zip(params.keys(), experiment))

        train_model(config)
