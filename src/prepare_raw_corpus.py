import os
import requests
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import random
from tqdm import tqdm

def download_from_url(url, filepath):
    r = requests.get(url)
    f = open(filepath, 'wb')
    f.write(r.content)
    f.close()
        
def save_to_local_disk(data_save_path):
    ## MSC
    dataset = load_dataset("gonced8/multi-session_chat")
    for split, dataset in dataset.items():
        dataset.to_json(f"{data_save_path}/multi_session_chat/{split}.json")
    
    ## ConversationChronicles
    dataset = load_dataset("jihyoung/ConversationChronicles")
    for split, dataset in dataset.items():
        dataset.to_json(f"{data_save_path}/conversation_chronicles/{split}.json")
        
    ## Carecall_memory
    os.makedirs(f"{data_save_path}/carecall_memory", exist_ok=True)
    download_from_url('https://raw.githubusercontent.com/naver-ai/carecall-memory/master/data/carecall-memory_en_auto_translated.json', \
                      f"{data_save_path}/carecall_memory/train.json")
    

def upload_to_huggingface(dataset: pd.DataFrame, save_name:str):
    raw_train = Dataset.from_pandas(dataset['train'])
    raw_valid = Dataset.from_pandas(dataset['validation'])
    raw_test = Dataset.from_pandas(dataset['test'])
    concat_dataset = DatasetDict({'train': raw_train, 'validation': raw_valid, 'test': raw_test})
    concat_dataset.push_to_hub(save_name)


class PrepareDialogueCorpus:
    def __init__(self):
        self.datasets=["multi_session_chat", "conversation_chronicles"]
    
    def prepare_dataset(self, dataset_name, dataset_path:str) -> pd.DataFrame:
        dataset = None
        
        if dataset_name=="multi_session_chat":
            dataset = self._prepare_MSC(dataset_path)
        elif dataset_name=="conversation_chronicles":
            dataset = self._prepare_ConversationChronicle(dataset_path)
        else:
            "Dataset name is not contained within the class."
        return dataset
    
    
    def _prepare_MSC(self, dataset_path:str) -> pd.DataFrame:
        print(dataset_path)
        df = pd.read_json(dataset_path, lines=True)

        multi_session = []
        for idx in range(len(df)): # number of data
            for sess in range(len(df['sessions'][idx])): # number of session
                multi_session_personas1 = df['sessions'][idx][sess]['personas'][0]['text']
                multi_session_personas2 = df['sessions'][idx][sess]['personas'][1]['text']

                multi_session_dialogue = []
                multi_session_speaker = []

                # number of turn
                for turn in range(len(df['sessions'][idx][sess]['dialogue'])):
                    multi_session_dialogue.append(df['sessions'][idx][sess]['dialogue'][turn]['text'])
                    multi_session_speaker.append(df['sessions'][idx][sess]['dialogue'][turn]['speaker'])

                multi_session.append(['MSC', idx, sess, multi_session_personas1, multi_session_personas2, multi_session_dialogue, multi_session_speaker])

        data = pd.DataFrame(multi_session, columns=['dataset', 'dialoug_id', 'session_id', 'persona1', 'persona2', 'dialogue', 'speaker'])
        return data

    def _prepare_ConversationChronicle(self, dataset_path:str) -> pd.DataFrame:
        print(dataset_path)
        df = pd.read_json(dataset_path, lines=True)
        df = df.sample(frac=0.05, random_state=2023).reset_index()

        multi_session = []
        for idx in range(len(df)): # number of data
            column_name = ["first", "second", "third", "fourth", "fifth"]
            for sess_num, c_name in enumerate(column_name): # number of session
                data_id = df['dataID'][idx]
                relationship = df['relationship'][idx]
                time_interval = df['time_interval'][idx][sess_num]
                summarization = df['summary'][idx][sess_num]
                dialogue = df[f'{c_name}_session_dialogue'][idx]
                speaker = df[f'{c_name}_session_speakers'][idx]
                multi_session.append(['CC', data_id, idx, sess_num, relationship, time_interval, summarization, dialogue, speaker])

        data = pd.DataFrame(multi_session, columns=['dataset', 'data_id', 'dialogue_id', 'session_id', 'relationship', 'time_interval', 'summarization', 'dialogue', 'speaker'])
        return data


class DialogueRelateTaskGenerator:
    def __init__(self):
        self.tasks=["MT_DG", "MT_DS", "MT_DHG", "MT_SRE", "MT", "MT_DG_SFT"]
    
    def run_task_generate(self, task_name:str, preprocessed_dataset:pd.DataFrame):
        
        if task_name=="MT_DG":
            dataset = self._task_dialouge_generation(preprocessed_dataset)
        elif task_name=="MT_DS":
            dataset = self._task_dialouge_summarization(preprocessed_dataset)
        elif task_name=="MT_DHG":
            dataset = self._task_dialouge_history_generation(preprocessed_dataset)
        elif task_name=="MT_SRE":
            dataset = self._task_speaker_relation_extraction(preprocessed_dataset)
        elif task_name=="MT2":
            dataset = self._task_mt(preprocessed_dataset)
        elif task_name=="MT_DG_SFT":
            dataset = self._task_mt_dg_sft(preprocessed_dataset)
        else:
            "Task name is not contained within the class."
        return dataset

    def _task_dialouge_generation(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df[df['time_interval']=="Start"].reset_index()
        
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            multi_turn_dialogue = []
            n_turn = len(df['dialogue'][idx])

            for turn in range(n_turn):
                row = f"{df['speaker'][idx][turn]}: {df['dialogue'][idx][turn]}\n"
                multi_turn_dialogue.append(row)

            rand_idx = random.randint(2, turn)
            multi_turn_dialogue_part = multi_turn_dialogue[:rand_idx]

            last_response = multi_turn_dialogue_part[-1]
            last_spaker = multi_turn_dialogue_part[-1].split(':')[0]
            multi_turn_dialogue_part[-1] = last_spaker + ': ###\n'
            context = ''.join(multi_turn_dialogue_part)

            prompt = f"""You will be shown a {len(multi_turn_dialogue_part)} turn dialogues between {df['speaker'][idx][0]} and {df['speaker'][idx][1]}. Please read and understand given Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""
            main_context = "```\nDialogue Session:\n" + context + "```\n\n"
            task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Session, please create an appropriate response.\n```\n\nTask Result:"""

            input = prompt + main_context + task_introduction
            output = last_response
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])
    
    def _task_dialouge_summarization(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df[df['time_interval']=="Start"].reset_index()
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            multi_turn_dialogue = []
            n_turn = len(df['dialogue'][idx])

            for turn in range(n_turn):
                row = f"{df['speaker'][idx][turn]}: {df['dialogue'][idx][turn]}\n"
                multi_turn_dialogue.append(row)

            context = ''.join(multi_turn_dialogue)
            prompt = f"""You will be shown a {len(multi_turn_dialogue)} turn dialogues between {df['speaker'][idx][0]} and {df['speaker'][idx][1]}. Please read and understand given Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""
            main_context = "```\nDialogue Session:\n" + context + "```\n\n"
            task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Session, please summarize the following Dialogue Session.\n```\n\nTask Result:"""

            input = prompt + main_context + task_introduction
            output = f"Dialogue Session Summary: {df['summarization'][idx]}"
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])

    def _task_dialouge_history_generation(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df[df['time_interval']=="Start"].reset_index()
        
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            multi_turn_dialogue = []
            n_turn = len(df['dialogue'][idx])

            for turn in range(n_turn):
                row = f"{df['speaker'][idx][turn]}: {df['dialogue'][idx][turn]}\n"
                multi_turn_dialogue.append(row)

            rand_idx = random.randint(0, turn)
            dialogue_span = multi_turn_dialogue[rand_idx]
            multi_turn_dialogue[rand_idx] = multi_turn_dialogue[rand_idx].split(':')[0] + ': ###\n'
            context = ''.join(multi_turn_dialogue)

            prompt = f"""You will be shown a {n_turn} turn dialogues between {df['speaker'][idx][0]} and {df['speaker'][idx][1]}. Please read and understand given Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""
            main_context = "```\nDialogue Session:\n" + context + "```\n\n"
            task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Session, please create an appropriate dialogue in the parts marked ###.\n```\n\nTask Result:"""

            input = prompt + main_context + task_introduction
            output = dialogue_span
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])

    def _task_speaker_relation_extraction(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
            df = preprocessed_dataset
            df = df[df['time_interval']=="Start"].reset_index()
            
            task_dsg = []
            speaker_list = ["Speaker 1", "Speaker 2"]
            for idx in tqdm(range(len(df))): # number of data
                multi_turn_dialogue = []
                n_turn = len(df['dialogue'][idx])

                for turn in range(n_turn):
                    speaker = speaker_list[turn%2]
                    row = f"{speaker}: {df['dialogue'][idx][turn]}\n"
                    multi_turn_dialogue.append(row)

                context = ''.join(multi_turn_dialogue)

                prompt = f"""You will be shown a {len(multi_turn_dialogue)} turn dialogues between {speaker_list[0]} and {speaker_list[1]}. Please read and understand given Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""
                main_context = "```\nDialogue Session:\n" + context + "```\n\n"
                task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Session, please guess the relationship between the two speakers.\n```\n\nTask Result:"""

                input = prompt + main_context + task_introduction
                output = f"Relationship: {df['relationship'][idx]}"
                task_dsg.append([input, output])
            return pd.DataFrame(task_dsg, columns=['input', 'output'])
    
    def _task_mt(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        # num_partitions = 4
        # rows_per_partitoins = len(df) // num_partitions
        # sub_df = [df.iloc[i:i+rows_per_partitoins] for i in range(0, len(df), rows_per_partitoins)]
        
        # d0 = self._task_dialouge_generation(sub_df[0])
        # d1 = self._task_dialouge_summarization(sub_df[1])
        # d2 = self._task_dialouge_history_generation(sub_df[2])
        # d3 = self._task_speaker_relation_extraction(sub_df[3])
        d0 = self._task_dialouge_generation(df)
        d1 = self._task_dialouge_summarization(df)
        d2 = self._task_dialouge_history_generation(df)
        d3 = self._task_speaker_relation_extraction(df)
        return pd.concat([d3, d2, d1, d0], ignore_index=True)

    def _task_mt_dg_sft(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df[df['time_interval']=="Start"].reset_index()
        
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            multi_turn_dialogue = []
            n_turn = len(df['dialogue'][idx])

            for turn in range(n_turn):
                row = f"{df['speaker'][idx][turn]}: {df['dialogue'][idx][turn]}\n"
                multi_turn_dialogue.append(row)

            rand_idx = random.randint(2, turn)
            multi_turn_dialogue_part = multi_turn_dialogue[:rand_idx]

            last_response = multi_turn_dialogue_part[-1]
            last_spaker = multi_turn_dialogue_part[-1].split(':')[0]
            multi_turn_dialogue_part[-1] = last_spaker + ': ###\n'
            context = ''.join(multi_turn_dialogue_part)

            # prompt = f"""You will be shown a {len(multi_turn_dialogue_part)} turn dialogues between {df['speaker'][idx][0]} and {df['speaker'][idx][1]}. Please read and understand given Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""
            main_context = "```\nDialogue Session:\n" + context + "```\n\n"
            # task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Session, please create an appropriate response.\n```\n\nTask Result:"""

            input = main_context
            output = last_response
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])

class MultiSessionTaskGenerator:
    def __init__(self):
        self.tasks = ["MS_DG", "MS_DS", "MS_DHG", "MS_TIE", "MS", "MS_DG_SFT"]
        
    def run_task_generate(self, task_name:str, preprocessed_dataset:pd.DataFrame): 
        if task_name=="MS_DG":
            dataset = self._task_dialouge_generation(preprocessed_dataset)
        elif task_name=="MS_DS":
            dataset = self._task_dialouge_summarization(preprocessed_dataset)
        elif task_name=="MS_DHG":
            dataset = self._task_dialouge_history_generation(preprocessed_dataset)
        elif task_name=="MS_TIE":
            dataset = self._task_time_interval_estimatoin(preprocessed_dataset)
        elif task_name=="MS2":
            dataset = self._task_ms(preprocessed_dataset)
        elif task_name=="MS_DG_SFT":
            dataset = self._task_ms_dg_sft(preprocessed_dataset)
        else:
            "Task name is not contained within the class."
        return dataset

    def _task_dialouge_generation(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df.sample(frac=0.05, random_state=2023).reset_index()
        
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            rand_sessions_num = random.randint(2, 5)
            column_name = ["first", "second", "third", "fourth", "fifth"]
            column_name = column_name[:rand_sessions_num]
            
            multi_sessoin_dialogue = []
            for sess_num, c_name in enumerate(column_name): # number of session
                
                if rand_sessions_num!=sess_num+1:
                    session = [f"{speaker}:{dialogue}\n" for speaker, dialogue in zip(df[f'{c_name}_session_speakers'][idx], df[f'{c_name}_session_dialogue'][idx])]
                    multi_sessoin_dialogue.append(''.join(session))
                else:
                    last_session = [f"{speaker}:{dialogue}\n" for speaker, dialogue in zip(df[f'{c_name}_session_speakers'][idx], df[f'{c_name}_session_dialogue'][idx])]
                    rand_idx = random.randint(2, len(last_session))
                    multi_turn_dialogue_part = last_session[:rand_idx]
                    
                    last_response = multi_turn_dialogue_part[-1]
                    last_spaker = multi_turn_dialogue_part[-1].split(':')[0]
                    multi_turn_dialogue_part[-1] = last_spaker + ': ###\n'
                    context = ''.join(multi_turn_dialogue_part)
                    multi_sessoin_dialogue.append(context)
            
            prompt = f"""You will be shown a {len(multi_sessoin_dialogue)} session dialogues between {df['first_session_speakers'][idx][0]} and {df['first_session_speakers'][idx][1]}. Please read and understand given multiple Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""
            
            main_context_list = [f"```\nDialogue Session #{idx+1}:\n" + context + "```\n\n" for idx, context in enumerate(multi_sessoin_dialogue)]
            main_context = ''.join(main_context_list)

            task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Sessions, please create an appropriate response.\n```\n\nTask Result:"""

            input = prompt + main_context + task_introduction
            output = last_response
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])

    def _task_dialouge_summarization(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df.sample(frac=0.05, random_state=2023).reset_index()
        
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            column_name = ["first", "second", "third", "fourth", "fifth"]
            
            multi_sessoin_dialogue = []
            for sess_num, c_name in enumerate(column_name): # number of session
                session = [f"{speaker}:{dialogue}\n" for speaker, dialogue in zip(df[f'{c_name}_session_speakers'][idx], df[f'{c_name}_session_dialogue'][idx])]
                multi_sessoin_dialogue.append(''.join(session))

            prompt = f"""You will be shown a {len(multi_sessoin_dialogue)} session dialogues between {df['first_session_speakers'][idx][0]} and {df['first_session_speakers'][idx][1]}. Please read and understand given multiple Dialogue Sessions, then complete the task under the guidance of Task Introduction.\n\n"""

            main_context_list = [f"```\nDialogue Session #{idx+1}:\n" + context + "```\n\n" for idx, context in enumerate(multi_sessoin_dialogue)]
            main_context = ''.join(main_context_list)

            task_introduction = f"""```\nTask Introduction:\nAfter reading the entire Dialogue Sessions, please summarize each of the following {len(multi_sessoin_dialogue)} Dialogue Sessions.\n```\n\nTask Result:"""
            input = prompt + main_context + task_introduction
            
            multi_session_summaries = df['summary'][idx]
            summary = [f"Dialogue Session Summary #{idx+1}:{summary}\n" for idx, summary in enumerate(multi_session_summaries)]
            output = ''.join(summary)
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])

    def _task_dialouge_history_generation(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df.sample(frac=0.05, random_state=2023).reset_index()
        
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            rand_sessions_num = random.randint(0, 2)
            column_name = ["first", "second", "third", "fourth", "fifth"]
            column_name = column_name[rand_sessions_num:rand_sessions_num+3]
            
            multi_sessoin_dialogue = []
            for sess_num, c_name in enumerate(column_name): # number of session
                if sess_num==1: # center session -> answer summarizatoin
                    summary = ''.join(df['summary'][idx])
                    multi_sessoin_dialogue.append("###")
                else: # session -> input dialogue
                    session = [f"{speaker}:{dialogue}" for speaker, dialogue in zip(df[f'{c_name}_session_speakers'][idx], df[f'{c_name}_session_dialogue'][idx])]
                    multi_sessoin_dialogue.append(''.join(session))
            
            prompt = f"""You will be shown a {len(multi_sessoin_dialogue)} session dialogues between {df['first_session_speakers'][idx][0]} and {df['first_session_speakers'][idx][1]}. Please read and understand given multiple Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""

            main_context_list = [f"```\nDialogue Session #{idx+1}:\n" + context + "```\n\n" for idx, context in enumerate(multi_sessoin_dialogue)]
            main_context = ''.join(main_context_list)

            task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Sessions, please create an appropriate Dialogue Session Summary in the parts marked ###.\n```\n\nTask Result:"""
            input = prompt + main_context + task_introduction
            
            output = ''.join(f"Dialogue Session Summary #{2}: {summary}")
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])
    
    def _task_time_interval_estimatoin(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df.sample(frac=0.05, random_state=2023).reset_index()
        
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            rand_sessions_num = random.randint(0, 3)
            column_name = ["first", "second", "third", "fourth", "fifth"]
            column_name = column_name[rand_sessions_num:rand_sessions_num+2]
            
            multi_sessoin_dialogue = []
            for sess_num, c_name in enumerate(column_name): # number of session
                session = [f"{speaker}:{dialogue}" for speaker, dialogue in zip(df[f'{c_name}_session_speakers'][idx], df[f'{c_name}_session_dialogue'][idx])]
                multi_sessoin_dialogue.append(''.join(session))

            time_interval = df['time_interval'][idx][rand_sessions_num+1]
                     
            prompt = f"""You will be shown a {len(multi_sessoin_dialogue)} session dialogues between {df['first_session_speakers'][idx][0]} and {df['first_session_speakers'][idx][1]}. Please read and understand given multiple Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""

            main_context_list = [f"```\nDialogue Session #{idx+1}:\n" + context + "```\n\n" for idx, context in enumerate(multi_sessoin_dialogue)]
            main_context = ''.join(main_context_list)

            task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Sessions, please guess the time interval between two dialogue sessions.\n```\n\nTask Result:"""
            input = prompt + main_context + task_introduction
            
            output = f"Time interval: {time_interval}"
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])
    
    def _task_ms(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        # num_partitions = 4
        # rows_per_partitoins = len(df) // num_partitions
        # sub_df = [df.iloc[i:i+rows_per_partitoins] for i in range(0, len(df), rows_per_partitoins)]
        
        # d0 = self._task_dialouge_generation(sub_df[0])
        # d1 = self._task_dialouge_summarization(sub_df[1])
        # d2 = self._task_dialouge_history_generation(sub_df[2])
        # d3 = self._task_time_interval_estimatoin(sub_df[3])
        d0 = self._task_dialouge_generation(df)
        d1 = self._task_dialouge_summarization(df)
        d2 = self._task_dialouge_history_generation(df)
        d3 = self._task_time_interval_estimatoin(df)
        return pd.concat([d3, d2, d1 ,d0], ignore_index=True) 

    def _task_ms_dg_sft(self, preprocessed_dataset:pd.DataFrame) -> pd.DataFrame:
        df = preprocessed_dataset
        df = df.sample(frac=0.05, random_state=2023).reset_index()
        
        task_dsg = []
        for idx in tqdm(range(len(df))): # number of data
            rand_sessions_num = random.randint(2, 5)
            column_name = ["first", "second", "third", "fourth", "fifth"]
            column_name = column_name[:rand_sessions_num]
            
            multi_sessoin_dialogue = []
            for sess_num, c_name in enumerate(column_name): # number of session
                
                if rand_sessions_num!=sess_num+1:
                    session = [f"{speaker}:{dialogue}\n" for speaker, dialogue in zip(df[f'{c_name}_session_speakers'][idx], df[f'{c_name}_session_dialogue'][idx])]
                    multi_sessoin_dialogue.append(''.join(session))
                else:
                    last_session = [f"{speaker}:{dialogue}\n" for speaker, dialogue in zip(df[f'{c_name}_session_speakers'][idx], df[f'{c_name}_session_dialogue'][idx])]
                    rand_idx = random.randint(2, len(last_session))
                    multi_turn_dialogue_part = last_session[:rand_idx]
                    
                    last_response = multi_turn_dialogue_part[-1]
                    last_spaker = multi_turn_dialogue_part[-1].split(':')[0]
                    multi_turn_dialogue_part[-1] = last_spaker + ': ###\n'
                    context = ''.join(multi_turn_dialogue_part)
                    multi_sessoin_dialogue.append(context)
            
            # prompt = f"""You will be shown a {len(multi_sessoin_dialogue)} session dialogues between {df['first_session_speakers'][idx][0]} and {df['first_session_speakers'][idx][1]}. Please read and understand given multiple Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"""
            
            main_context_list = [f"```\nDialogue Session #{idx+1}:\n" + context + "```\n\n" for idx, context in enumerate(multi_sessoin_dialogue)]
            main_context = ''.join(main_context_list)

            # task_introduction = """```\nTask Introduction:\nAfter reading the entire Dialogue Sessions, please create an appropriate response.\n```\n\nTask Result:"""

            input = main_context
            output = last_response
            task_dsg.append([input, output])
        return pd.DataFrame(task_dsg, columns=['input', 'output'])



if __name__=="__main__":
    
    download_path= '../data/downloads'
    task_save_path = '../data/tasks'
    huggingface_user_name='nayohan'
    dataset_names=["conversation_chronicles"]
    # task_list = ["MT_DG", "MT_DS", "MT_DHG", "MT_SRE"]
    # task_list =["MT_DG_SFT", "MT"]
    task_list = ["MT2"]
    # multi_sessoin_task_list = ["MS_DG", "MS_DS", "MS_DHG", "MS_TIE"]
    # multi_sessoin_task_list = ["MS_DG_SFT", "MS"]
    multi_sessoin_task_list = ["MS2"]

    # ## 1. Download dialogue datasets.
    # os.makedirs(download_path, exist_ok=True)
    # save_to_local_disk(download_path)

    # ## 2. Preprocess dialogue datset to same struture
    # corup_generator = PrepareDialogueCorpus()
    # for dataset_name in dataset_names:
    #     MSC={}
    #     for split in ['train', 'validation', 'test']:
    #         MSC[split] = corup_generator.prepare_dataset(dataset_name, f'{download_path}/{dataset_name}/{split}.json')
    #         MSC[split].to_json(f'{download_path}/{dataset_name}/prerocessed_{split}.jsonl', orient='records', lines=True)
    #     upload_to_huggingface(MSC, f'{huggingface_user_name}/{dataset_name}')
    
    # ## 3. Generate Multi-turn dialouge related instruction tasks. It's only generate last dataset_name task.
    # MSC = load_dataset(f'{huggingface_user_name}/conversation_chronicles')
    # os.makedirs(task_save_path, exist_ok=True)
    # task_generator = DialogueRelateTaskGenerator()
    # for task in task_list:
    #     os.makedirs(f'{task_save_path}/{task}', exist_ok=True)
    #     dataset_dict={}
    #     for split in ['train', 'validation', 'test']:
    #         dataset_dict[split] = task_generator.run_task_generate(task, pd.DataFrame(MSC[split]))
    #         dataset_dict[split].to_json(f'{task_save_path}/{task}/{split}.jsonl', orient='records', lines=True)
    #     # upload_to_huggingface(dataset_dict, f'{huggingface_user_name}/{task}')
        
    
    # ## 4. Multi-sessoin dialouge tasks
    # MSC = load_dataset('jihyoung/ConversationChronicles')
    # os.makedirs(task_save_path, exist_ok=True)
    # task_generator = MultiSessionTaskGenerator()
    # for task in multi_sessoin_task_list:
    #     os.makedirs(f'{task_save_path}/{task}', exist_ok=True)
    #     dataset_dict={}
    #     for split in ['train', 'validation', 'test']:
    #         dataset_dict[split] = task_generator.run_task_generate(task, pd.DataFrame(MSC[split]))
    #         dataset_dict[split].to_json(f'{task_save_path}/{task}/{split}.jsonl', orient='records', lines=True)
            
            
    ## 5. Multi-turn/sessoin dialouge tasks ALL
    MT = load_dataset(f'{huggingface_user_name}/conversation_chronicles')
    MS = load_dataset('jihyoung/ConversationChronicles')
    os.makedirs(task_save_path, exist_ok=True)
    
    task = "ALL"
    mt_task_generator = DialogueRelateTaskGenerator()
    ms_task_generator = MultiSessionTaskGenerator()
    os.makedirs(f'{task_save_path}/{task}', exist_ok=True)
    dataset_dict={}
    for split in ['train', 'validation', 'test']:
        mt_dataset = mt_task_generator.run_task_generate("MT2", pd.DataFrame(MT[split]))
        ms_dataset = ms_task_generator.run_task_generate("MS2", pd.DataFrame(MS[split]))
        dataset_dict[split] = pd.concat([mt_dataset, ms_dataset])
        dataset_dict[split].to_json(f'{task_save_path}/{task}/{split}.jsonl', orient='records', lines=True)