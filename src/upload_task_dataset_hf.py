
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

# dg = load_dataset('/home/uj-user/Yo/hybrid-ltm/data/tasks/DG')
# dg.push_to_hub('nayohan/DialogueGeneration')

dg = load_dataset('/home/uj-user/Yo/hybrid-ltm/data/tasks/DHG2')
dg.push_to_hub('nayohan/DialogueHistoryGeneration')
