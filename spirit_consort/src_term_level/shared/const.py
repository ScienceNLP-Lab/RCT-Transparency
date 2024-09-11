task_ner_labels = {
    'consort': [
        '1a_Title_Randomized', '1b_Title_Type', '1c_Title_Framework', '1d_Title_Centers',
        '1e_Title_Population', '1f_Title_Intervention', '1g_Title_Acronym',
        '3a_Registry_number', '8a_Design_Type', '8b_Design_Framework',
        '8c_Design_Centers', '8d_Design_Ratio',
        '14a_Sample_size', '16a_Randomization_Generation',
        '16b_Randomization_Type', '16c_Randomization_Block_size', '16d_Randomization_Strata',
        '17a_Masking_People_masked', '17b_Masking_Not_masked',
        '17c_Masking_Type', '20c_Statistical_methods_Analysis_population',
        '20d_Statistical_methods_Missing_data'
    ]
}

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label

def get_ner_labelmap(label_list):
    label2id_entity = {}
    id2label_entity = {}
    label2id_trigger = {}
    id2label_trigger = {}
    for i, label in enumerate(label_list):
        if label.isupper():
            label2id_trigger[label] = i + 1
            id2label_trigger[i + 1] = label
        else:
            label2id_entity[label] = i + 1
            id2label_entity[i + 1] = label
    return label2id_entity, id2label_entity, label2id_trigger, id2label_trigger
