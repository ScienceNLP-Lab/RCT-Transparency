## Input data format for term extraction task
In the **processed_data** folder, we follow the protocol of the original PURE paper to construct the input: each line of the input file contains one document.

```bash
{
  # PMCID (please make sure doc_key can be used to identify a certain document)
  "doc_key": "PMC6814122",

  # sentences in the document, each sentence is a list of tokens
  "sentences": [
    ["Implementing", "integrated", "services", "in", "routine", ...],
    [...],
    [...],
    ["The", "remaining", "49", "agencies", "were", "assigned", ...]
    ...
  ],

  # entities (boundaries and entity type) in each sentence
  "ner": [
    [[0, 2, "1f_Title_Intervention"], [4, 7, "1e_Title_Population"], ...], 
    [...],
    [...],
    [[1494, 1495, "8c_Design_Centers"]] # span indices are article-level
    ...,
  ],

  # section headers (hierarchical, from outer to inner.)
  "section_headers": [
    ["Title"], 
    [...],
    [...],
    [["Methods", "Participants"]]
    ...,
  ],
}
```
