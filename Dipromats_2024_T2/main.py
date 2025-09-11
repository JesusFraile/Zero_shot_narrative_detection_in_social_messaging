

# %%
import os
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import torch
import json
import time
import re
from tqdm import tqdm
import numpy as np
import bitsandbytes as bnb
#%%
import argparse


def main():
    # RUTE_LLMS='/data/jfraile/llms'
    # RUTE_FILES='/data/jfraile/Programs'

    RUTE_LLMS='/mnt/data/jfraile/llms'
    RUTE_FILES='/mnt/data/jfraile/Programs'
    model_paths={'mixtral': f'{RUTE_LLMS}/Mixtral-8x7B-Instruct-v0.1',
                'calme': f'{RUTE_LLMS}/calme-2.4-rys-78b-quant-4bits', 
                'calme_A30': f'{RUTE_LLMS}/calme-2.4-rys-78b-quant-4bits_A30',
                'granite':f'{RUTE_LLMS}/granite-3.3-2b-instruct',
                'exaone':f'{RUTE_LLMS}/EXAONE-3.5-7.8B-Instruct',
                'gemma':f'{RUTE_LLMS}/gemma-3-12b-it',
                'calme_gguf': f'{RUTE_LLMS}/calme-2.4-rys-78b-GGUF',
                'qwen_14B': f'{RUTE_LLMS}/Qwen2.5-14B', 
                'qwen': f'{RUTE_LLMS}/Qwen2.5-72B-Instruct-quant-4bits', 
                'deepseek_qwen': f'{RUTE_LLMS}/DeepSeek-R1-Distill-Qwen-14B', 
                'deepseek_llama': f'{RUTE_LLMS}/DeepSeek-R1-Distill-Llama-70B'}
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help=f'Models aviliables: {model_paths.keys()}')
    parser.add_argument('--model_description', required=True)
    parser.add_argument('--approach_description', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--lang')
    args=parser.parse_args()
    model_name=args.model
    model_description=args.model_description
    approach_description=args.approach_description
    run=int(args.run)
    lang=args.lang
    
    approach=f'{model_name}_zero_shot_{model_description}_{approach_description}_{run}'
    print('='*10)
    print(approach)
    print('='*10)


    if lang in ['es', 'en']:
        print(f"Language fixed -> {lang}")
        languages=[lang]
    else:
        print(f"NO Language fixed")
        languages=['es', 'en']


    if model_description!='no':
        if model_description=='original':
            descriptions=pd.read_json(f'{RUTE_FILES}/Narratives_Identification/narratives_descriptions/original_description.json')
        else:
            descriptions=pd.read_json(f'{RUTE_FILES}/Narratives_Identification/narratives_descriptions/zero_shot_description/z_s_{model_description}_{approach_description}.json')
        



    def get_description(narrative_id):
        if model_description=='no':
            return ''
        else:
            text=descriptions[descriptions['narrative']==narrative_id]['result'].iloc[0]
            return text


    class model():
        def __init__(self, model_path):
            if 'Mixtral' == model_name:
                
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                    torch_dtype=torch.bfloat16).eval()
            elif 'qwen_14B' == model_name:
                              
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                    torch_dtype=torch.bfloat16).eval()

            elif 'calme' == model_name:

                bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',             
                                                                quantization_config=bnb_config).eval()
            elif 'calme_A30' == model_name:

                bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',             
                                                                quantization_config=bnb_config).eval()
            elif 'calme_gguf' == model_name:

                gguf_file_name=f'{RUTE_LLMS}/calme-2.4-rys-78b-GGUF/calme-2.4-rys-78b.Q4_K_M.gguf'
                self.model= AutoModelForCausalLM.from_pretrained(model_path, gguf_file=gguf_file_name, device_map='auto')
            
            elif 'granite' == model_name:
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                    torch_dtype=torch.bfloat16).eval()
            elif 'exaone' == model_name:
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                    torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
            elif 'gemma' == model_name:
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()

            elif 'Qwen2.5-72B' == model_name:
                bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', quantization_config=bnb_config).eval()
            
            elif 'DeepSeek-R1-Distill-Qwen-32B' == model_name:
                bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.bfloat16
            )
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                                 quantization_config=bnb_config).eval()
                
            elif 'DeepSeek-R1-Distill-Llama-70B' == model_name:
                bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model= AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                                 quantization_config=bnb_config).eval()
                
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            self.pipeline=pipeline(
                "text-generation",
                model=self.model,
                torch_dtype=torch.float16,
                device_map="auto",
                tokenizer=self.tokenizer)
            
            self.pipeline_kwargs= dict(
                        do_sample=True,
                        temperature=0.75,
                        top_k=5,
                        num_return_sequences=1,
                        max_new_tokens=50,
                        return_full_text=False,
                        use_cache=True)    
        def get_tokenizer(self):
            return self.tokenizer

                
        def inference(self, text):
            # Tokenizer input text
            # inputs = self.tokenizer(text, return_tensors="pt")
            
            # Doing inference
            self.pipeline_kwargs['eos_token_id']=self.tokenizer.eos_token_id
            for out in self.pipeline(text,**self.pipeline_kwargs):
                model_out=out['generated_text']
            
            return model_out
        




    def CH_Agent(lang, tweet):
        prompt_agent=f"""
                Your primary role is to analyze a tweet {tweet} and categorize them according to predefined narrative themes that reflect different portrayals and perspectives of China.
                Your classification should help in understanding the overarching sentiments and strategic messaging in public discourse.
                Narratives to Detect:

            1:  The West as Immoral and Hostile.
                {get_description('CH1')}
            2:  China as a Benevolent Power.
                {get_description('CH2')}
            3:  China’s Epic History.
                {get_description('CH3')}
            4:  China’s Political System and Values.
                {get_description('CH4')}
            5:  Success of the Chinese Communist Party’s Government.
                {get_description('CH5')}
            6:  China’s Cultural, Natural, and Heritage Appeal.
                {get_description('CH6')}
        
            Instructions for Classification:

            1- Read carefully the tweet {tweet}
            2- Determine which narrative(s) it supports based on the content and sentiment expressed, a tweet may align with at most 2 narratives if it incorporates elements from more than one category. 
            It is possible that the tweet does not support any narrative. 
            3- You have to generate a Json structure:

                    {{
                        "classification": [1, 2, 3, 4, 5, 6] at most 2 categories or [] if the tweet doesn't support any narrative,
                        "reasoning": 'reasoning of the answer in maximum 50 words.'
                    }}.

                """
        return prompt_agent

    def RU_Agent(lang, tweet):
        
        prompt_agent=f"""
                You are to analyze and categorize a tweet {tweet} based on predefined Russian narrative themes.
                This classification is intended to help identify how Russia is portrayed and how it positions itself in global discourse through social media.
                Narratives to Detect:

        1: The West as Immoral and Hostile.
            {get_description('RU1')}
        2: Russia Leads an Alternative System.
            {get_description('RU2')}
        3:  Russia as a Benevolent Partner.
            {get_description('RU3')}
        4: Russian History is Admirable.
            {get_description('RU4')}
        5:  Russia at the World’s Forefront.
            {get_description('RU5')}
        6:  Russia's Cultural, Natural, and Heritage Appeal.
            {get_description('RU6')}

        Instructions for Classification:

            1- Read carefully the tweet {tweet}
            2- Determine which narrative(s) it supports based on the content and sentiment expressed, a tweet may align with at most 2 narratives if it incorporates elements from more than one category. 
            It is possible that the tweet does not support any narrative. 
            3- You have to generate a Json structure:

                    {{
                        "classification": [1, 2, 3, 4, 5, 6] at most 2 categories or [] if the tweet doesn't support any narrative,
                        "reasoning": 'reasoning of the answer in maximum 50 words.'
                    }}.

                    """
        return prompt_agent

    def EU_Agent(lang, tweet):

        prompt_agent=f"""
                Your role is to analyze and categorize a {tweet} according to predefined narrative themes related to the portrayal of the European Union.
                This will aid in understanding how the EU is represented in public discussions on social media.
                Narratives to Detect:

        1: The European Union is a Successful Story.
            {get_description('EU1')}
        2: The European Union is United.
            {get_description('EU2')}
        3:  The European Union is Useful for Its Citizens.
            {get_description('EU3')}
        4: The European Union as an Avant-Garde Political Actor.
            {get_description('EU4')}
        5: The European Union as a Global Champion of Fair Causes.
            {get_description('EU5')}
        6: The European Union Contributes to a Better World.
            {get_description('EU6')}

        Instructions for Classification:

            1- Read carefully the tweet {tweet}
            2- Determine which narrative(s) it supports based on the content and sentiment expressed, a tweet may align with at most 2 narratives if it incorporates elements from more than one category. 
            It is possible that the tweet does not support any narrative. 
            3- You have to generate a Json structure:

                    {{
                        "classification": [1, 2, 3, 4, 5, 6] at most 2 categories or [] if the tweet doesn't support any narrative,
                        "reasoning": 'reasoning of the answer in maximum 50 words.'
                    }}.

                    """
        return prompt_agent


    def US_Agent(lang, tweet):
        
        prompt_agent=f"""
                Your task is to analyze and categorize a tweet {tweet} according to predefined narrative themes that reflect various portrayals of the United States.
                This classification will contribute to understanding the representation of the U.S. in public discourse on social media.
                Narratives to Detect:

        1: The US has an Honorable Tradition.
            {get_description('US1')}
        2: The US has an Admirable Society.
            {get_description('US2')}
        3: The US has Hostile Enemies.
            {get_description('US3')}
        4:  The US is a Force for Good.
            {get_description('US4')}
        5: The US is a Great International Ally.
            {get_description('US5')}
        6:  The US is at the Forefront.
            {get_description('US6')}
        
        Instructions for Classification:

            1- Read carefully the tweet {tweet}
            2- Determine which narrative(s) it supports based on the content and sentiment expressed, a tweet may align with at most 2 narratives if it incorporates elements from more than one category. 
            It is possible that the tweet does not support any narrative. 
            3- You have to generate a Json structure:

                    {{
                        "classification": [1, 2, 3, 4, 5, 6] at most 2 categories or [] if the tweet doesn't support any narrative,
                        "reasoning": 'reasoning of the answer in maximum 50 words.'
                    }}.

                    """
        return prompt_agent


    def extract_json_content(text):

        def extract_reasoning(text):
            # Search for the reasoning value
            match_reasoning = re.search(r'"reasoning":\s*\'(.*?)\'', text)
            if match_reasoning:
                # Extract and return the reasoning text
                return match_reasoning.group(1).strip()  # Remove any leading/trailing whitespace
            return None  # Return None if nothing is found

        def extract_classification(text):
            # Search for the values of "classification"
            match_classification = re.search(r'"classification":\s*\[(.*?)\]', text)
            if match_classification:
                # Extract the numbers inside the brackets
                classification_values = match_classification.group(1).strip()  # Remove any whitespace
                if classification_values:  # Check if it is not empty
                    return list(map(int, classification_values.split(',')))  # Convert to integers
                else:
                    return []  # Return an empty list if there are no values
            return None  # Return None if nothing is found
        return extract_classification(text), extract_reasoning(text)
    
    def check_nan(value):
        try:
            if value==None:
                return True
            if np.isnan(float(value)):
                return True
            return False
        except (ValueError, TypeError):
            return False
        
    class EvaluationProcessPipeline():
        def __init__(self, model, row):
            self.model=model
            self.row=row
            self.country = self.row['country']
            self.text = self.row["text"]
            self.language=self.row['language']
            self.date=self.row['UTC'][:10]
            self.user=self.row['username']
            self.country_code = ''  # Placeholder for country code
            if self.country == 'USA':
                self.prompt = US_Agent(self.language, self.text)
                self.country_code = 'US'
            elif self.country == 'EU':
                self.prompt = EU_Agent(self.language, self.text)
                self.country_code = 'EU'
            elif self.country == 'Russia':
                self.prompt = RU_Agent(self.language, self.text)
                self.country_code = 'RU'
            elif self.country == 'China':
                self.prompt = CH_Agent(self.language, self.text)
                self.country_code = 'CH'
            else:
                # Raise an error if the country does not match any known option
                raise ValueError(f"No valid agent found for the specified country: {self.country}")
            
        def process_tweet(self):
            max_attempts=5
            attempts=0

            while attempts < max_attempts:
                try:
                    chat=[{'role':'user', 'content':self.prompt}]
                    
                    final_prompt=self.model.get_tokenizer().apply_chat_template(chat, tokenize=False)
                    entry_result=self.model.inference(final_prompt)

                    # print("Raw entry_result:", entry_result)  # Debugging line to see what is being returned
                    # Attempt to parse the JSON data
                    entry_result_json=extract_json_content(entry_result)
                    
                    label_numbers = entry_result_json[0]
                    # Transform the list of numbers into a list of formatted strings with the country code
                    labels = [f"{self.country_code}{num}" for num in label_numbers]
                    print("Extracted labels:", labels)  # Debugging line to confirm label extraction
                    reasoning=entry_result_json[1]
                    return labels, reasoning  # Successful parsing and return, exit loop
                
                except (json.JSONDecodeError, Exception) as error:
                    # Handle both JSON parsing errors and other exceptions
                    print(f"Error encountered (attempt {attempts + 1}):", error)
                    print(entry_result)
                    attempts += 1
                    # time.sleep(1)  # Sleep before retrying
            print("Failed after", max_attempts, "attempts.")
            return [], entry_result


                

        


    
    inf_model=model(model_paths[model_name])


    
    for lang in languages:
        print(f"Starting {lang} process")
        path_to_save=f'{RUTE_FILES}/Narratives_Identification/paper_runs/models_results/{approach}/test_{lang}.pkl'
        if os.path.exists(path_to_save):
            df_path=path_to_save
            df_t2=pd.read_pickle(df_path)
            print("Loaded DataFrame")
            time.sleep(5)
        else:
            print("New DataFrame")
            time.sleep(5)
            df_path=f'{RUTE_FILES}/datasets/DIPROMATS_2024_T2/test_dataset_{lang}.json'
            with open(df_path) as json_data:
                data=json.load(json_data)
                df_t2=pd.json_normalize(data)
                df_t2.reset_index(inplace=True)
                df_t2['country'].replace({'Russia':'Russia', 'China':'China', 'European Union':'EU', 'USA':'USA'}, inplace=True)
            df_t2['language']=lang
            df_t2['narratives']=pd.Series()
            df_t2['reasoning']=pd.Series()
        col_list=['language', 'id', 'country', 'tweet_id', 'narratives']

        
        for i in tqdm(range(len(df_t2))):
            narratives=df_t2['narratives'].iloc[i]
            if check_nan(narratives):
                print("====================")
                print(f'{lang} -> {i+1}/{len(df_t2)}')
                label, reason = EvaluationProcessPipeline(inf_model, df_t2.iloc[i]).process_tweet()
                df_t2.at[i, 'narratives'] = label
                df_t2.at[i, 'reasoning'] = reason
                os.makedirs(f'{RUTE_FILES}/Narratives_Identification/paper_runs/models_results/{approach}', exist_ok=True)
                df_t2[col_list].to_json(f'{RUTE_FILES}/Narratives_Identification/paper_runs/models_results/{approach}/test_{lang}.json', orient='records', indent=2, force_ascii=False)
                df_t2.to_pickle(f'{RUTE_FILES}/Narratives_Identification/paper_runs/models_results/{approach}/test_{lang}.pkl')

if __name__ == "__main__":
    main()

