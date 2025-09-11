
# %%
import os
# from matplotlib.pyplot import get_figlabels
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, 
    pipeline
)
import torch
import json
import time
import re
from tqdm import tqdm
import numpy as np
#%%
import argparse

# RUTE_LLMS='/data/jfraile/llms'
# RUTE_FILES='/data/jfraile/Programs'

RUTE_LLMS='/mnt/data/jfraile/llms'
RUTE_FILES='/mnt/data/jfraile/Programs'

def main():
    # import subprocess
    # import threading
    # import time

    # def log_gpu_usage(logfile="gpu_usage.log", interval=5):
    #     with open(logfile, "a") as f:
    #         while True:
    #             f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    #             f.write(subprocess.getoutput("nvidia-smi") + "\n")
    #             f.flush()
    #             time.sleep(interval)

    # t = threading.Thread(target=log_gpu_usage, daemon=True)
    # t.start()

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
    parser.add_argument('--lang', required=True)
    parser.add_argument('--run', required=True)
    
    args=parser.parse_args()
    run=int(args.run)
    model_name=args.model
    model_description=args.model_description
    approach_description=args.approach_description
    lang=args.lang

    
    description_type='zero_shot_desc'
    selected_set='test_set'
    approach=f'{model_name}_zero_shot_{model_description}_{approach_description}_all_in_one_{run}'
    description_approach=f'{model_description}_z_s_desc_{approach_description}'

    print("="*10)
    print(approach)
    print("="*10)

    df_cc=pd.read_csv(f'{RUTE_FILES}/datasets/SemEval_2025_T10_S2/narratives_CC.txt', sep=';', header=None)
    df_cc.rename(columns={0:'id', 1:'narrative'}, inplace=True)
    df_urw=pd.read_csv(f'{RUTE_FILES}/datasets/SemEval_2025_T10_S2/narratives_URW.txt', sep=';', header=None)
    df_urw.rename(columns={0:'id', 1:'narrative'}, inplace=True)
    df_narratives=pd.concat([df_urw, df_cc], ignore_index=True)

    if model_description!='no':
        if model_description=='original':
            df_desc=pd.read_json(f'{RUTE_FILES}/datasets/SemEval_2025_T10_S2/titles_and_descriptions.json')
        else:
            df_desc=pd.read_json(f'{RUTE_FILES}/SemEval_2025_T10_S2/description_generation/{description_type}/{description_approach}_v2.json')  




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
                                                    torch_dtype=torch.bfloat16).eval()

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
            
        def get_tokenizer(self):
            return self.tokenizer

                
        def inference(self, text, pipeline_kwargs):
            # Tokenizer input text
            # inputs = self.tokenizer(text, return_tensors="pt")
            
            # Doing inference
            self.pipeline_kwargs=pipeline_kwargs
            self.pipeline_kwargs['eos_token_id']=self.tokenizer.eos_token_id
            for out in self.pipeline(text,**self.pipeline_kwargs):
                model_out=out['generated_text']
            
            return model_out
        


    def get_description(narr):
        if model_description=='no':
            return ''
        else:
            return df_desc[df_desc['id']==narr]['description'].iloc[0]
        
    def prompt(news):
        news_red=news[:2000]
        prompt_agent=f"""
        Your primary role is to analyze a new and categorize them according to predefined narrative and sub-narrative themes that reflect different portrayals and perspectives of the Ukraine-Russia war (URW) or Climate Change (CC).
        Your classification should help in understanding the overarching sentiments and strategic messaging in public discourse.

        Narratives and Sub-narratives to Detect:

        URW1.: Blaming the war on others rather than the invader. 
            URW1.1: Ukraine is the aggressor. {get_description('URW1.1')}
            URW1.2: The West are the aggressors. {get_description('URW1.2')}
        URW2.: Discrediting Ukraine.
            URW2.1: Rewriting Ukraine’s history. {get_description('URW2.1')}
            URW2.2: Discrediting Ukrainian nation and society. {get_description('URW2.2')}
            URW2.3: Discrediting Ukrainian military. {get_description('URW2.3')}
            URW2.4: Discrediting Ukrainian government and officials and policies. {get_description('URW2.4')}
            URW2.5: Ukraine is a puppet of the West. {get_description('URW2.5')}
            URW2.6: Ukraine is a hub for criminal activities. {get_description('URW2.6')}
            URW2.7: Ukraine is associated with nazism. {get_description('URW2.7')}
            URW2.8: Situation in Ukraine is hopeless. {get_description('URW2.8')}
        URW3.: Russia is the Victim.
            URW3.1: The West is russophobic. {get_description('URW3.1')}
            URW3.2: Russia actions in Ukraine are only self-defence. {get_description('URW3.2')}
            URW3.3: UA is anti-RU extremists. {get_description('URW3.3')}
        URW4.: Praise of Russia.
            URW4.1: Praise of Russian military might. {get_description('URW4.1')}
            URW4.2: Praise of Russian President Vladimir Putin. {get_description('URW4.2')}
            URW4.3: Russia is a guarantor of peace and prosperity. {get_description('URW4.3')}
            URW4.4: Russia has international support from a number of countries and people. {get_description('URW4.4')}
            URW4.5: Russian invasion has strong national support. {get_description('URW4.5')}
        URW5.: Overpraising the West.
            URW5.1: NATO will destroy Russia. {get_description('URW5.1')}
            URW5.2: The West belongs in the right side of history. {get_description('URW5.2')}
            URW5.3: The West has the strongest international support. {get_description('URW5.3')}
        URW6.: Speculating war outcomes.
            URW6.1: Russian army is collapsing. {get_description('URW6.1')}
            URW6.2: Russian army will lose all the occupied territories. {get_description('URW6.2')}
            URW6.3: Ukrainian army is collapsing. {get_description('URW6.3')}
        URW7.: Discrediting the West, Diplomacy.
            URW7.1: The EU is divided. {get_description('URW7.1')}
            URW7.2: The West is weak. {get_description('URW7.2')}
            URW7.3: The West is overreacting. {get_description('URW7.3')}
            URW7.4: The West does not care about Ukraine, only about its interests. {get_description('URW7.4')}
            URW7.5: Diplomacy does/will not work. {get_description('URW7.5')}
            URW7.6: West is tired of Ukraine. {get_description('URW7.6')}
        URW8.: Negative Consequences for the West.
            URW8.1: Sanctions imposed by Western countries will backfire. {get_description('URW8.1')}
            URW8.2: The conflict will increase the Ukrainian refugee flows to Europe. {get_description('URW8.2')}
        URW9.: Distrust towards Media.
            URW9.1: Western media is an instrument of propaganda. {get_description('URW9.1')}
            URW9.2: Ukrainian media cannot be trusted. {get_description('URW9.2')}
        URW10.: Amplifying war-related fears.
            URW10.1: By continuing the war we risk WWIII. {get_description('URW10.1')}
            URW10.2: Russia will also attack other countries. {get_description('URW10.2')}
            URW10.3: There is a real possibility that nuclear weapons will be employed. {get_description('URW10.3')}
            URW10.4: NATO should/will directly intervene. {get_description('URW10.4')}
        URW11.: Hidden plots by secret schemes of powerful groups. {get_description('URW11.')}

        CC1.: Criticism of climate policies.
            CC1.1: Climate policies are ineffective. {get_description('CC1.1')}
            CC1.2: Climate policies have negative impact on the economy. {get_description('CC1.2')}
            CC1.3: Climate policies are only for profit. {get_description('CC1.3')}
        CC2.: Criticism of institutions and authorities.
            CC2.1: Criticism of the EU. {get_description('CC2.1')}
            CC2.2: Criticism of international entities. {get_description('CC2.2')}
            CC2.3: Criticism of national governments. {get_description('CC2.3')}
            CC2.4: Criticism of political organizations and figures. {get_description('CC2.4')}
        CC3.: Climate change is beneficial.
            CC3.1: CO2 is beneficial. {get_description('CC3.1')}
            CC3.2: Temperature increase is beneficial. {get_description('CC3.2')}
        CC4.: Downplaying climate change.
            CC4.1: Climate cycles are natural. {get_description('CC4.1')}
            CC4.2: Weather suggests the trend is global cooling. {get_description('CC4.2')}
            CC4.3: Temperature increase does not have significant impact. {get_description('CC4.3')}
            CC4.4: CO2 concentrations are too small to have an impact. {get_description('CC4.4')}
            CC4.5: Human activities do not impact climate change. {get_description('CC4.5')}
            CC4.6: Ice is not melting. {get_description('CC4.6')}
            CC4.7: Sea levels are not rising. {get_description('CC4.7')}
            CC4.8: Humans and nature will adapt to the changes. {get_description('CC4.8')}
        CC5.: Questioning the measurements and science.
            CC5.1: Methodologies/metrics used are unreliable/faulty. {get_description('CC5.1')}
            CC5.2: Data shows no temperature increase. {get_description('CC5.2')}
            CC5.3: Greenhouse effect/carbon dioxide do not drive climate change. {get_description('CC5.3')}
            CC5.4: Scientific community is unreliable. {get_description('CC5.4')}
        CC6.: Criticism of climate movement.
            CC6.1: Climate movement is alarmist. {get_description('CC6.1')}
            CC6.2: Climate movement is corrupt. {get_description('CC6.2')}
            CC6.3: Ad hominem attacks on key activists. {get_description('CC6.3')}
        CC7.: Controversy about green technologies.
            CC7.1: Renewable energy is dangerous. {get_description('CC7.1')}
            CC7.2: Renewable energy is unreliable. {get_description('CC7.2')}
            CC7.3: Renewable energy is costly. {get_description('CC7.3')}
            CC7.4: Nuclear energy is not climate friendly. {get_description('CC7.4')}
        CC8.: Hidden plots by secret schemes of powerful groups.
            CC8.1: Blaming global elites. {get_description('CC8.1')}
            CC8.2: Climate agenda has hidden motives. {get_description('CC8.2')}
        CC9.: Amplifying Climate Fears.
            CC9.1: Earth will be uninhabitable soon. {get_description('CC9.1')}
            CC9.2: Amplifying existing fears of global warming. {get_description('CC9.2')}
            CC9.3: Doomsday scenarios for humans. {get_description('CC9.3')}
            CC9.4: Whatever we do it is already too late. {get_description('CC9.4')}
        CC10.: Green policies are geopolitical instruments.
            CC10.1: Climate-related international relations are abusive/exploitative. {get_description('CC10.1')}
            CC10.2: Green activities are a form of neo-colonialism. {get_description('CC10.2')}

        The text of the news item you have to analyse is:

        (Start of news item to be analysed)

        {news_red}

        (End of news item to be analysed)

        Instructions for Classification:

        1- Read carefully the news.
        2- Determine which sub-narrative(s) it supports based on the content and sentiment expressed, a news item can align with several sub-narratives if it incorporates elements from more than one category. 
        If the text supports a narrative, e.g. URW1., but does not support any of the sub-narratives proposed for that narrative you have to write the code of the narrative followed by OTH, e.g. URW1.OTH
        If the text does not support any narrative write OTH.OTH

        Valid labels are: ['URW1.1', 'URW1.2', 'URW1.OTH', 'URW2.1', 'URW2.2', 'URW2.3', 'URW2.4', 'URW2.5', 'URW2.6', 'URW2.7', 'URW2.8', 'URW2.OTH', 'URW3.1', 'URW3.2', 'URW3.3', 'URW3.OTH', 'URW4.1', 'URW4.2', 'URW4.3', 'URW4.4', 'URW4.5', 'URW4.OTH', 'URW5.1', 'URW5.2', 'URW5.3', 'URW5.OTH', 'URW6.1', 'URW6.2', 'URW6.3', 'URW6.OTH', 'URW7.1', 'URW7.2', 'URW7.3', 'URW7.4', 'URW7.5', 'URW7.6', 'URW7.OTH', 'URW8.1', 'URW8.2', 'URW8.OTH', 'URW9.1', 'URW9.2', 'URW9.OTH', 'URW10.1', 'URW10.2', 'URW10.3', 'URW10.4', 'URW10.OTH', 'URW11.OTH', 'CC1.1', 'CC1.2', 'CC1.3', 'CC1.OTH', 'CC2.1', 'CC2.2', 'CC2.3', 'CC2.4', 'CC2.OTH', 'CC3.1', 'CC3.2', 'CC3.OTH', 'CC4.1', 'CC4.2', 'CC4.3', 'CC4.4', 'CC4.5', 'CC4.6', 'CC4.7', 'CC4.8', 'CC4.OTH', 'CC5.1', 'CC5.2', 'CC5.3', 'CC5.4', 'CC5.OTH', 'CC6.1', 'CC6.2', 'CC6.3', 'CC6.OTH', 'CC7.1', 'CC7.2', 'CC7.3', 'CC7.4', 'CC7.OTH', 'CC8.1', 'CC8.2', 'CC8.OTH', 'CC9.1', 'CC9.2', 'CC9.3', 'CC9.4', 'CC9.OTH', 'CC10.1', 'CC10.2', 'CC10.OTH', 'OTH.OTH']
        
        3- You have to generate a .json structure:

                {{
                    "classification": ['URW1.1', 'URW2.1', 'CC4.OTH', ...] The valid labels of the sub-narratives supported by the text according to instruction 2.,
                    "reasoning": 'reasoning of the answer in maximum 50 words.'
                }}.
        """
        
        return prompt_agent

    def extract_json_content(text):
        allowed_classifications=['URW1.1', 'URW1.2', 'URW1.OTH', 'URW2.1', 'URW2.2', 'URW2.3', 'URW2.4', 'URW2.5', 'URW2.6', 'URW2.7', 'URW2.8', 'URW2.OTH', 'URW3.1', 'URW3.2', 'URW3.3', 'URW3.OTH', 'URW4.1', 'URW4.2', 'URW4.3', 'URW4.4', 'URW4.5', 'URW4.OTH', 'URW5.1', 'URW5.2', 'URW5.3', 'URW5.OTH', 'URW6.1', 'URW6.2', 'URW6.3', 'URW6.OTH', 'URW7.1', 'URW7.2', 'URW7.3', 'URW7.4', 'URW7.5', 'URW7.6', 'URW7.OTH', 'URW8.1', 'URW8.2', 'URW8.OTH', 'URW9.1', 'URW9.2', 'URW9.OTH', 'URW10.1', 'URW10.2', 'URW10.3', 'URW10.4', 'URW10.OTH', 'URW11.OTH', 'CC1.1', 'CC1.2', 'CC1.3', 'CC1.OTH', 'CC2.1', 'CC2.2', 'CC2.3', 'CC2.4', 'CC2.OTH', 'CC3.1', 'CC3.2', 'CC3.OTH', 'CC4.1', 'CC4.2', 'CC4.3', 'CC4.4', 'CC4.5', 'CC4.6', 'CC4.7', 'CC4.8', 'CC4.OTH', 'CC5.1', 'CC5.2', 'CC5.3', 'CC5.4', 'CC5.OTH', 'CC6.1', 'CC6.2', 'CC6.3', 'CC6.OTH', 'CC7.1', 'CC7.2', 'CC7.3', 'CC7.4', 'CC7.OTH', 'CC8.1', 'CC8.2', 'CC8.OTH', 'CC9.1', 'CC9.2', 'CC9.3', 'CC9.4', 'CC9.OTH', 'CC10.1', 'CC10.2', 'CC10.OTH', 'OTH.OTH']
        def extract_reasoning(text):
            # Search for the reasoning value
            match_reasoning = re.search(r'"reasoning":\s*\'(.*?)\'', text)
            if match_reasoning:
                # Extract and return the reasoning text
                return match_reasoning.group(1).strip()  # Remove any leading/trailing whitespace
            return None  # Return None if nothing is found

        def extract_classification(text):
                match_classification = re.search(r'"classification":\s*\[(.*?)\]', text)
                if match_classification:
                    classification_values = re.findall(r'["\'](.*?)["\']', match_classification.group(1))
                    if all(value in allowed_classifications for value in classification_values):
                        return classification_values
                    else:
                        print("Some classification values are not allowed.")
                        return None  # Return None if any value is not allowed
                return None  
        return extract_classification(text), extract_reasoning(text)
    
    def check_nan(value):
        try:
            if np.isnan(float(value)):
                return True
            return False
        except (ValueError, TypeError):
            return False  

    class EvaluationProcessPipeline():
        def __init__(self, model, row):
            self.model=model
            self.row=row
            self.text = self.row["text"]
            self.language=self.row['lang']
            self.thematic = ''  # Placeholder for country code
            # self.narrative=[]
            self.pipeline_kwargs= dict(
        do_sample=True,
        temperature=0.75,
        top_k=5,
        num_return_sequences=1,
        max_new_tokens=50,
        return_full_text=False)

        
        def process_text(self):
            max_attempts=5
            attempts=0
            self.prompt=prompt(self.text)
            while attempts < max_attempts:
                try:
                    chat=[{'role':'user', 'content':self.prompt}]
                    
                    final_prompt=self.model.get_tokenizer().apply_chat_template(chat, tokenize=False)
                    entry_result=self.model.inference(final_prompt, self.pipeline_kwargs)

                    # print("Raw entry_result:", entry_result)  # Debugging line to see what is being returned
                    # Attempt to parse the JSON data
                    entry_result_json=extract_json_content(entry_result)
                    
                    labels = entry_result_json[0]
                    # print(labels)
                    
                    reasoning=entry_result_json[1]
                    return labels, reasoning  # Successful parsing and return, exit loop
                
                except (json.JSONDecodeError, Exception) as error:
                    # Handle both JSON parsing errors and other exceptions
                    print(f"Error encountered (attempt {attempts + 1}):", error)
                    attempts += 1
                    # time.sleep(1)  # Sleep before retrying
            print("Failed after", max_attempts, "attempts.")
            return [], entry_result
        
               

        


    
    inf_model=model(model_paths[model_name])
 
    print(f"Starting {lang} process")
    
    path_to_save=f'{RUTE_FILES}/SemEval_2025_T10_S2/paper_runs/zero_shot_all_in_one/models_results/{approach}/{selected_set}_{lang}.pkl'

    if os.path.exists(path_to_save):
        df_path=path_to_save
        df_t2=pd.read_pickle(df_path)
        print("Loaded DataFrame")
        time.sleep(5)
    else:
        df_t2=pd.DataFrame(columns=['file_name', 'text', 'lang'])
        text_files=os.listdir(f'{RUTE_FILES}/datasets/SemEval_2025_T10_S2/{selected_set}/{lang}/subtask-2-documents/')
        if lang!='EN':
            text_files=[name for name in text_files if 'translated' in name]
        for text_file in tqdm(text_files):
            text_path=f'{RUTE_FILES}/datasets/SemEval_2025_T10_S2/{selected_set}/{lang}/subtask-2-documents/{text_file}'
            f=open(text_path, 'r')
            text=f.read()
            row={'file_name': text_file, 'text': text, 'lang': lang}
            df_t2=pd.concat([df_t2, pd.DataFrame([row])], ignore_index=True)

            
        df_t2['pred_subnarratives']=pd.Series()


    for i in tqdm(range(len(df_t2))):
        subnarratives=df_t2['pred_subnarratives'].iloc[i]
        if check_nan(subnarratives):
            print("====================")
            print(f'{lang} -> {i+1}/{len(df_t2)}')
            pipe=EvaluationProcessPipeline(inf_model, df_t2.iloc[i])
            #Thematic Clasification
            subnarrative_list, reason = pipe.process_text()
            
            print(f'Lista subnarrativas -> {subnarrative_list}')
            df_t2.at[i, 'pred_subnarratives'] = subnarrative_list

            os.makedirs(f'{RUTE_FILES}/SemEval_2025_T10_S2/paper_runs/zero_shot_all_in_one/models_results/{approach}', exist_ok=True)
            
            df_t2.to_pickle(f'{RUTE_FILES}/SemEval_2025_T10_S2/paper_runs/zero_shot_all_in_one/models_results/{approach}/{selected_set}_{lang}.pkl')
            
if __name__ == "__main__":
    main()

