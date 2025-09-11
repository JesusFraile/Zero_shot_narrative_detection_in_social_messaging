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
    approach=f'{model_name}_zero_shot_{model_description}_{approach_description}_3_steps_{run}'
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
        


    def subnarrative_prompt(narrative):

        if model_description!='no':
            dft=df_narratives[df_narratives['id'].str.startswith(narrative)]
            text=''
            for i in range(1,len(dft)):
                # print(f'{narrative}.{i}')
                t=df_desc[df_desc['id']==f'{narrative}{i}']['description'].iloc[0]
                text=f"""{text}
                {i}. {dft['narrative'].iloc[i]}. {t}
    """
            return text
        if model_description=='no':
            dft=df_narratives[df_narratives['id'].str.startswith(narrative)]
            text=''
            for i in range(1,len(dft)):
                text=f"""{text}
                {i}. {dft['narrative'].iloc[i]}.
    """
            return text
    def narrative_description_prompt(narrative):
        if model_description!='no':
            t=df_desc[df_desc['id']==narrative]['description'].iloc[0]
            return {t}
        else:
            return ''



    def thematic_classification_prompt(news):
        news_red=news[:2000]
        prompt_agent=f"""
            Your main function is to analyse a news item and classify it according to the thematic of the text. 
            Themes to detect:

            1: The war between Ukraine and Russia.
            2: Climate change

            The text of the news item you have to analyse is:

            (Start of news item to be analysed)

            {news_red}

            (End of news item to be analysed)

            Instructions for Classification:

            1- Read carefully the news.
            2- Determine what the main topic of the news item is.
            3- You have to generate a .json structure:

                    {{
                        "classification": [1, 2] only one of the two categories (Write it between []),
                        "reasoning": 'reasoning of the answer in maximum 50 words.'
                    }}.

                """
        return prompt_agent
    
    def narratives_classification_prompt(news, thematic):
        news_red=news[:2000]
        if thematic=='URW':
            prompt_agent=f"""
                Your primary role is to analyze a new and categorize them according to predefined narrative themes that reflect different portrayals and perspectives of the Russia-Ukraine war.
                Your classification should help in understanding the overarching sentiments and strategic messaging in public discourse.
                Narratives to Detect:

                1: Blaming the war on others rather than the invader. {narrative_description_prompt('URW1.')}
                2: Discrediting Ukraine. {narrative_description_prompt('URW2.')}
                3: Russia is the Victim. {narrative_description_prompt('URW3.')}
                4: Praise of Russia. {narrative_description_prompt('URW4.')}
                5: Overpraising the West. {narrative_description_prompt('URW5.')}
                6: Speculating war outcomes. {narrative_description_prompt('URW6.')}
                7: Discrediting the West, Diplomacy. {narrative_description_prompt('URW7.')}
                8: Negative Consequences for the West. {narrative_description_prompt('URW8.')}
                9: Distrust towards Media. {narrative_description_prompt('URW9.')}
                10: Amplifying war-related fears. {narrative_description_prompt('URW10.')}
                11: Hidden plots by secret schemes of powerful groups. {narrative_description_prompt('URW11.')}

                The text of the news item you have to analyse is:

                (Start of news item to be analysed)

                {news_red}

                (End of news item to be analysed)

                Instructions for Classification:

                1- Read carefully the news.
                2- Determine which narrative(s) it supports based on the content and sentiment expressed, a news item can align with several narratives if it incorporates elements from more than one category. 
                It is possible that the news does not support any narrative. 
                3- You have to generate a .json structure:

                        {{
                            "classification": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] or [] if the tweet doesn't support any narrative (YOU CAN ONLY WRITE NUMBERS),
                            "reasoning": 'reasoning of the answer in maximum 50 words.'
                        }}.

                    """
        elif thematic=='CC':
            prompt_agent=f"""
                Your primary role is to analyze a new and categorize them according to predefined narrative themes that reflect different portrayals and perspectives of the Climate Change.
                Your classification should help in understanding the overarching sentiments and strategic messaging in public discourse.
                Narratives to Detect:

                1: Criticism of climate policies. {narrative_description_prompt('CC1.')}
                2: Criticism of institutions and authorities. {narrative_description_prompt('CC2.')}
                3: Climate change is beneficial. {narrative_description_prompt('CC3.')}
                4: Downplaying climate change. {narrative_description_prompt('CC4.')}
                5: Questioning the measurements and science. {narrative_description_prompt('CC5.')}
                6: Criticism of climate movement. {narrative_description_prompt('CC6.')}
                7: Controversy about green technologies. {narrative_description_prompt('CC7.')}
                8: Hidden plots by secret schemes of powerful groups. {narrative_description_prompt('CC8.')}
                9: Amplifying Climate Fears. {narrative_description_prompt('CC9.')}
                10: Green policies are geopolitical instruments. {narrative_description_prompt('CC10.')}
                

                The text of the news item you have to analyse is:

                (Start of news item to be analysed)

                {news_red}

                (End of news item to be analysed)

                Instructions for Classification:

                1- Read carefully the news.
                2- Determine which narrative(s) it supports based on the content and sentiment expressed, a news item can align with several narratives if it incorporates elements from more than one category. 
                It is possible that the news does not support any narrative. 
                3- You have to generate a .json structure:

                        {{
                            "classification": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] or [] if the tweet doesn't support any narrative (YOU CAN ONLY WRITE NUMBERS),
                            "reasoning": 'reasoning of the answer in maximum 50 words.'
                        }}.


                    """
        else:
            print("Theme Error in narrative_classification_prompt")
        return prompt_agent

    def subnarratives_classification_prompt(news, narrative_id):
        news_red=news[:2000]
        if narrative_id!='URW11.':
            t=df_narratives[df_narratives['id']==narrative_id]['narrative'].iloc[0]
            dft=df_narratives[df_narratives['id'].str.startswith(narrative_id)]
            l=len(dft)
            l=list(range(1,l))
            if 'CC' in narrative_id:
                theme='Climate Change'
            elif 'URW' in narrative_id:
                theme='Russia-Ukraine war'
            prompt_agent=f"""
                Your primary role is to analyze a new and categorize them according to predefined narrative themes that reflect different portrayals and perspectives of the {theme}.
                The news item supports that: {t}.
                Your classification should help in understanding the overarching sentiments and strategic messaging in public discourse.
                Narratives to Detect:

                {subnarrative_prompt(narrative_id)}

                The text of the news item you have to analyse is:

                (Start of news item to be analysed)

                {news_red}

                (End of news item to be analysed)

                Instructions for Classification:

                1- Read carefully the news.
                2- Determine which narrative(s) it supports based on the content and sentiment expressed, a news item can align with several narratives if it incorporates elements from more than one category. 
                It is possible that the news does not support any narrative. 
                3- You have to generate a .json structure:

                        {{
                            "classification": {l} or [] if the tweet doesn't support any narrative (Write it between []) (YOU CAN ONLY WRITE NUMBERS),
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
                    return list(map(lambda x: int(x.strip().strip('"').strip("'")), classification_values.split(',')))  # Convert to integers
                else:
                    return []  # Return an empty list if there are no values
            return None  # Return None if nothing is found
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

        def get_figlabels(self, label_numbers, task_type, narrative=None):
            if task_type=='thematic':
                try:
                    if len(label_numbers)>0:
                        if label_numbers[0]==1:
                            return 'URW'
                        elif label_numbers[0]==2:
                            return 'CC'
                        else:
                            return 'ERROR'
                    else:
                        return 'ERROR'
                except:
                    return 'ERROR'
            elif task_type=='narrative':
                valid_labels=[f'CC{i}.' for i in range(1, 11)]+[f'URW{i}.' for i in range(1, 12)]
                if len(label_numbers)==0:
                    return ['OTH.OTH']
                else:
                    labels = [f"{self.thematic}{num}." for num in label_numbers]
                    if all(label in valid_labels for label in labels):
                        return labels
                    else:
                        return 'ERROR'
            elif task_type=='subnarrative':
                if len(label_numbers)==0:
                    return [f"{narrative}OTH"]
                else:
                    labels = [f"{narrative}{num}" for num in label_numbers]
                    return labels


        def process_text(self, task_type, thematic=None, narrative=None):
            max_attempts=5
            attempts=0
            if task_type=='thematic':
                self.prompt=thematic_classification_prompt(self.text)
            elif task_type=='narrative':
                self.prompt=narratives_classification_prompt(self.text, thematic)
            elif task_type=='subnarrative':
                self.prompt=subnarratives_classification_prompt(self.text, narrative)
            while attempts < max_attempts:
                try:
                    chat=[{'role':'user', 'content':self.prompt}]
                    
                    final_prompt=self.model.get_tokenizer().apply_chat_template(chat, tokenize=False)
                    entry_result=self.model.inference(final_prompt, self.pipeline_kwargs)

                    # print("Raw entry_result:", entry_result)  # Debugging line to see what is being returned
                    # Attempt to parse the JSON data
                    entry_result_json=extract_json_content(entry_result)
                    
                    label_numbers = entry_result_json[0]
                    # print(label_numbers)
                    labels=self.get_figlabels(label_numbers, task_type, narrative)
                    if (labels == 'ERROR') or (labels==None):
                        attempts += 1
                        continue  
                    # print("Extracted labels:", labels)  # Debugging line to confirm label extraction
                    reasoning=entry_result_json[1]
                    return labels, reasoning  # Successful parsing and return, exit loop
                
                except (json.JSONDecodeError, Exception) as error:
                    # Handle both JSON parsing errors and other exceptions
                    print(f"Error encountered (attempt {attempts + 1}):", error)
                    attempts += 1
                    # time.sleep(1)  # Sleep before retrying
            print("Failed after", max_attempts, "attempts.")
            return ['OTH.OTH'], entry_result
        
        


    inf_model=model(model_paths[model_name])


    print(f"Starting {lang} process")
    path_to_save=f'{RUTE_FILES}/SemEval_2025_T10_S2/paper_runs/3_steps/models_results/{approach}/{selected_set}_{lang}.pkl'
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
            them_label, reason = pipe.process_text(task_type='thematic', thematic=None, narrative=None)
            pipe.thematic=them_label
            df_t2.at[i, 'pred_topic'] = them_label
            print(f'Lista temas -> {them_label}')

            if them_label==['OTH.OTH']:
                df_t2.at[i, 'pred_subnarratives'] = ['OTH.OTH']
                continue
            #Narratives Clasification
            
            narr_label, reason = pipe.process_text(task_type='narrative', thematic=them_label, narrative=None)
            if len(narr_label)==0:
                narr_label=['OTH.OTH']
            # if len(narr_label)==0:
            #      narr_label=['OTH.OTH']
            print(f'Lista narrativas -> {narr_label}')
            subnarrative_list=[]
            for narr in narr_label:
                #Narratives Clasification
                if narr in ('OTH.OTH', 'URW11.'):
                    if narr=='URW11.':
                        subnarrative_list.extend([f"{narr}OTH"])
                    else:
                        subnarrative_list.extend(['OTH.OTH'])
                else:
                    subnarr_label, reason = pipe.process_text(task_type='subnarrative', thematic=them_label, narrative=narr)
                    subnarrative_list.extend(subnarr_label)
            print(f'Lista subnarrativas -> {subnarrative_list}')
            df_t2.at[i, 'pred_subnarratives'] = subnarrative_list




            os.makedirs(f'{RUTE_FILES}/SemEval_2025_T10_S2/paper_runs/3_steps/models_results/{approach}', exist_ok=True)
            
            df_t2.to_pickle(f'{RUTE_FILES}/SemEval_2025_T10_S2/paper_runs/3_steps/models_results/{approach}/{selected_set}_{lang}.pkl')

if __name__ == "__main__":
    main()

