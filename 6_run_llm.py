from hypersearch import prepare_data
from itertools import product
from typing import List
import pandas as pd
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from tqdm.auto import tqdm
import random
import os
import litellm
from copy import deepcopy
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

load_dotenv(find_dotenv())

#LLM's tags, llm's will responde with this exact text as an output
TAG_FAKE = "FAKE NEWS"
TAG_TRUE = "VERDADEIRO"
TAG_INVALID = "INVALID" #for invalid llm responses

#Helper for the .parquet files
LABEL_FAKE = "fake"
LABEL_TRUE = "true"

#Search text config, only applies for '_extra' datasets
SEARCH_MAX_SIZE = 1 # Max size of search results for each sample
SEARCH_ADD_LINK = True # Add the link to the search text
SEARCH_REDUCE_VERBOSITY = False # Verbosity of the search text, I recommend True for BERT and False for LLM's
SEARCH_FILTER_DOMAIN = False

#SYSTEM PROMPT for the simple case, without search results
PROMPT_SIMPLE = (
    "Abaixo contém textos de mensagens e notícias em português, "
    "sua tarefa é classificar se o texto contém uma Fake News ou Não.\n"
    f"Responda apenas com as tags \"{TAG_FAKE}\" ou \"{TAG_TRUE}\"."
)

#SYSTEM_PROMPT for the extra case, with search results
PROMPT_EXTRA = (
    "Abaixo contém textos de mensagens e notícias em português, "
    "sua tarefa é classificar se o texto contém uma Fake News ou Não.\n"
    "Para axiliar na classificação também será também fornecido um contexto extra "
    "que corresponde a uma busca no google pelos termos do texto a ser classificado.\n"
    f"Responda apenas com  as tags \"{TAG_FAKE}\" ou \"{TAG_TRUE}\"."
)


def format_query(text: str, context: str=None) -> str:
    if context is not None:
        raise Exception("Invalid")
    return (
        f"---Texto a ser classificado:---\n{text}\n---\n"
        f"Classifique se o texto acima é verdadeiro com a tag '{TAG_TRUE}' ou se contem fake news com a tag "
        f"'{TAG_FAKE}'. Responda apenas com a tag."
    )

def format_query_extra(text: str, context: str) -> str:
    return (
        f"---Contexto Extra---\n{context}\n"
        f"---Texto a ser classificado---\n{text}\n---\n"
        f"Classifique se o texto acima é verdadeiro com a tag '{TAG_TRUE}' ou se contem fake news com a tag "
        f"'{TAG_FAKE}'. Responda apenas com a tag."
    )

PROMPT_TYPE_HELPER = {
    "simple": {
        "prompt": PROMPT_SIMPLE,
        "format_query": format_query
    },
    "extra": {
        "prompt": PROMPT_EXTRA,
        "format_query": format_query_extra
    }
}


def prepare_data_llm(config: dict, version: str) -> List[pd.DataFrame]:
    train, dev, test = prepare_data(
        config,
        search_max_size = SEARCH_MAX_SIZE,
        search_add_link = SEARCH_ADD_LINK,
        search_reduce_verbosity = SEARCH_REDUCE_VERBOSITY,
        search_filter_domains = SEARCH_FILTER_DOMAIN,
        search_filter_social_media = version == "no_social_media",
    )

    #Sample balanced few_shot
    fewshot_items = []
    fewshot_size = config['fewshot']
    
    round_robin = None
    shuffled_train = train.sample(frac=1, random_state=42)
    for i, row in shuffled_train.iterrows():
        if row['labels'] != round_robin:
            fewshot_items.append(row)
            round_robin = row['labels']
        if len(fewshot_items) == fewshot_size:
            break
    
    fewshot_df = pd.DataFrame(fewshot_items)
    fewshot_df = fewshot_df.sample(frac=1, random_state=42)

    return fewshot_df, test
    

def preprare_fewshot_messages_llm(fewshot_df, has_extra_context=False):
    data_type = 'simple' if not has_extra_context else 'extra'
    system_prompt = PROMPT_TYPE_HELPER[data_type]["prompt"]
    format_query_func = PROMPT_TYPE_HELPER[data_type]["format_query"]

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    for i, row in fewshot_df.iterrows():
        if 'text_a' in row:
            user_content = format_query_func(row['text_a'], row['text_b'])
        else:
            user_content = format_query_func(row['text'])

        messages.append({
            'role': 'user',
            'content': user_content
        })
        messages.append({
            'role': 'assistant',
            'content': TAG_FAKE if row['labels'] == LABEL_FAKE else TAG_TRUE
        })

    return messages

def run_model(config: dict, version:str):
    logger.debug(f"Running config: {str(config)}")
    
    experiment_name = "_".join(
        str(p).replace('.', '-').replace('/', '-') for p in config.values()
    )+ '_' + version

    logger.debug(f"Experiment name: {experiment_name}")

    output_dir = os.path.join("output", "llm_camera_ready", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    fewshot_df, test = prepare_data_llm(config, version)

    logger.debug("Fewshot set")
    logger.debug(fewshot_df.head(3).to_dict("records"))

    logger.debug("Test set")
    logger.debug(test.head(3).to_dict("records"))

    logger.debug(f"fewshot_df size: {len(fewshot_df)} / test size: {len(test)}")

    has_extra_context = 'extra' in config['data']
    data_type = 'simple' if not has_extra_context else 'extra'

    logger.debug(f"Running data_type: {data_type} / has_extra_context: {has_extra_context}")

    base_messages = preprare_fewshot_messages_llm(fewshot_df, has_extra_context)
    format_query_func = PROMPT_TYPE_HELPER[data_type]["format_query"]

    few_shot_message_debug = ""
    for m in base_messages:
        few_shot_message_debug += f"### Role: {m['role']}\nMessage:\n"+m['content']+"\n\n"
    few_shot_message_debug = few_shot_message_debug[:2000] + '\n\n{...}\n\n' + few_shot_message_debug[-2000:]
    logger.debug(f"Few-shot messages:\n{few_shot_message_debug}")

    os.makedirs(output_dir, exist_ok=True)
    requests_results_file = os.path.join(output_dir, 'requests_results.jsonl')

    logger.debug(f"requests_results_file file: {requests_results_file}")

    requests_results = []
    if os.path.exists(requests_results_file):
        #Load existing result data
        logger.info(f"requests_results_file exists, loading...")
        requests_results = []
        with open(requests_results_file, 'r') as f:
            for line in f:
                requests_results.append(json.loads(line))

    index_already_requested = set()
    for r in requests_results:
        index_already_requested.add(r['dataset_index'])
    
    if len(index_already_requested) > 0:
        logger.info(f"{len(index_already_requested)}/{len(test)} rows already processed, skipping")

    print("Running queries")
    count = 0
    for i, row in test.iterrows():
        if i in index_already_requested:
            count += 1
            continue

        row_tag_label = TAG_FAKE if row['labels'] == LABEL_FAKE else TAG_TRUE

        if 'text_a' in row:
            user_content = format_query_func(row['text_a'], row['text_b'])
        else:
            user_content = format_query_func(row['text'])

        messages = deepcopy(base_messages)
        messages.append(
            {
                'role': 'user',
                'content': user_content
            }
        )

        if count % 30 == 0:
            logger.debug(f"Running id:{i} - {count}/{len(test)}: {messages[-1]}")

        response = litellm.completion(
            model=config['model'], 
            messages=messages,
            temperature=0.0,
            drop_params=True,
            seed=config["seed"],
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
        )

        if count % 30 == 0:
            logger.debug(f"Response id:{i} - {count}/{len(test)}: {response.json()}")
        
        response_text = response.choices[0].message.content

        true_label_index = response_text.find(TAG_TRUE)
        fake_label_index = response_text.find(TAG_FAKE)

        response_tag_label = TAG_INVALID
        if fake_label_index >= 0 or true_label_index >= 0:
            if true_label_index == -1:
                response_tag_label = TAG_FAKE
            elif fake_label_index == -1:
                response_tag_label = TAG_TRUE
            else:
                #Uses the tag that came first in the response
                response_tag_label = TAG_FAKE if fake_label_index < true_label_index else TAG_TRUE

        response_label = None
        if response_tag_label == TAG_FAKE:
            response_label = LABEL_FAKE
        if response_tag_label == TAG_TRUE:
            response_label = LABEL_TRUE

        response_data = {
            'order': count,
            #config
            'model': config['model'],
            'config': config,
            #dataset info
            'dataset_index': i,
            'dataset_row': row.to_dict(),
            'dataset_row_label': row['labels'],
            'dataset_row_tag_label': row_tag_label,
            #request response info
            'request': messages,
            'response_raw': response.json(),
            'response_text': response_text,
            'response_label': response_label,
            'response_tag_label': response_tag_label,
            #result:
            'is_correct': response_label == row['labels']
        }

        requests_results.append(response_data)

        if count % 30 == 0:
            logger.debug(f"Response label:{i} - {count}/{len(test)}: {response_label}")

        with open(requests_results_file, 'a', encoding='utf8') as f:
            f.write(json.dumps(response_data, ensure_ascii=False) + '\n')
        
        count += 1

    labels = [LABEL_TRUE, LABEL_FAKE]
    y_true = [str(r['dataset_row_label']) for r in requests_results]
    y_pred = [str(r['response_label']) for r in requests_results]

    print('y_true', y_true)
    print('y_pred', y_pred)

    cls_report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    cls_report_print = classification_report(y_true, y_pred, labels=labels)
    cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)

    res_data = {
        "classification_report": cls_report,
        "classification_report_print": cls_report_print,
        "confusion_matrix": cf_matrix.tolist(),
        "accuracy": accuracy
    }

    result_json = os.path.join(output_dir, 'result.json')
    with open(result_json, 'w') as f:
        json.dump(res_data, f, indent=2, ensure_ascii=False)

    logger.debug(f"Finished config: {str(config)}")
    logger.debug(f"Results {result_json}:\n{cls_report_print}")


if __name__ == "__main__":

    params = {
        "model": ["gemini/gemini-1.5-flash"],
        "data": ["processed_extra", "processed"],
        "dataset": [
            "Fake.br",
            "COVID19.BR"
        ],
        "fewshot": [15],
        "version": ["social_media", "no_social_media"],
        "seed": [1, 2, 3]
    }

    available_experiments = []
    for experiment in product(*params.values()):
        available_experiments.append(experiment)

    logger.debug(f"Grid-Search: {len(available_experiments)} runs")

    random.shuffle(available_experiments)
    for experiment in tqdm(available_experiments):
        config = dict(zip(params.keys(), experiment))
        version = config.pop("version")

        run_model(config, version=version)
        #break
