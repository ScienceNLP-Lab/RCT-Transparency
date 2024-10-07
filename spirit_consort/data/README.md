# Data Documentation

We provide SPIRIT CONSORT dataset in this data dictionary. 
This dictionary includes the annotation guideline, all 100 pairs of articles, and the document-level, sentence-level, and term-level datasets.

---
- You can find the guidelines for annotating all 83 items [here](https://osf.io/ha73p).

- **documents/** dictionary comprises 100 protocol articles and their 100 corresponding results articles.

- **articles.csv** contains annotations at the article level, indicating whether each article reports on a specific item or not. Each row represents a single article.
  
> | Column name | Description | 
> | :---: | :---: |
> | Protocol/Results | Whether the article is a Protocol or Results paper |
> | PairID | Index for the article pair in the dataset |
> | PMCID | PubMed Central ID of the article |
> | ChecklistItem | The specific checklist item being evaluated |
> | Reported | Whether the checklist item is reported in the article (1 = reported / 0 = not reported) |
> | Split | The split to which the article belongs (e.g., train/test/valid) |

- **sentences.csv** contains annotations at the sentence level, indicating which checklist items each sentence describes. Each row corresponds to a single sentence from an article.

> | Column name | Description | 
> | :---: | :---: |
> | Protocol/Results | Whether the article is a Protocol or Results paper |
> | PairID | Index for the article pair in the dataset |
> | PMCID | PubMed Central ID of the article |
> | SentenceID | Index for the sentence within the article |
> | Sentence | Full text of the sentence |
> | SentenceNoMarkers| Sentence text without Section header identifiers(#) |
> | ChecklistItem | The checklist item(s) that the sentence describes |
> | SectionHeaders | The section(s) to which the sentence belongs |
> | IsSectionHeader | Whether the sentence is a section header (1 = Yes / 0 = No) |
> | SentenceStartOffset | The starting character position of the sentence in the article |
> | SentenceEndOffset | The ending character position of the sentence in the article |
> | Split | The split to which the sentence belongs (e.g., train/test/valid) |

- The folder **terms** contains term-level annotations with two sub-directories: 
    - **raw_data** contains two different zip files (Protocol and Results) which compress original data files in brat standoff annotation format.
    - **processed_data** contains three JSON files corresponding to training, validation, and test splits, and all of them follow the the SciERC dataset format [Luan et al., 2018](https://aclanthology.org/D18-1360/). 

    
    Each row in a JSON file corresponds to a single article that includes the following keys:

> | Key name | Description | 
> | :---: | :---: |
> | doc_key | PMID/PMCID for each article |
> | sentences | a list of tokens for each sentence |
> | ner | a list of terms in the article, including their token-level start and end offsets and the corresponding checklist item labels |
> | section_headers | all section headers for each sentence |

