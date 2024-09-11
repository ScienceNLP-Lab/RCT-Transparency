import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from train import calculate_multilabel_instance_metrics,\
    calculate_multilabel_article_metrics,\
    calculate_multilabel_section_metrics

SPIRIT = ['1a_Title_Randomized', '1b_Title_Type', '1e_Title_Population', '1f_Title_Intervention', '1g_Title_Acronym',
          '3a_Registry_number', '4_Funding', '5a_Sponsor', '5b_Contributors_roles', '5c_Oversight_committees',
          '7_Objectives', '8a_Design_Type', '8b_Design_Framework', '8c_Design_Centers', '8d_Design_Ratio', '9_Setting',
          '10a_Participants_inclusion', '10b_Center_interventionist_inclusion', '11a_Intervention_Description',
          '11b_Intervention_Modification', '11c_Intervention_Monitoring', '11d_Intervention_Concomitant',
          '12a_Outcomes_Definitions', '13_Participant_timeline', '14a_Sample_size', '14b_Sample_Calculation',
          '15_Recruitment', '16a_Randomization_Generation', '16b_Randomization_Type', '16c_Randomization_Block_size',
          '16d_Randomization_Strata', '16e_Allocation_Mechanism', '16f_Allocation_Concealment', '16g_Personnel_Sequence',
          '16h_Personnel_Enrollment', '17a_Masking_People_masked', '17b_Masking_Not_masked', '17c_Masking_Type',
          '17d_Masking_Unblinding', '17e_Masking_Similarity', '18a_Data_Collection', '18b_Data_Retention',
          '19_Data_Management', '20a_Statistical_methods_Outcomes', '20b_Statistical_methods_Other_Analyses',
          '20c_Statistical_methods_Analysis_population', '20d_Statistical_methods_Missing_data',
          '21a_Data_monitoring_committee', '21b_Interim_analyses', '21c_Stopping_guidelines', '22_Harms_non-systematic',
          '23_Auditing', '24_Ethics', '25_Amendments', '26a_Consent_Obtaining', '26b_Consent_Provisions',
          '27_Confidentiality', '28_Financial_interests', '29_Data_access', '30_Post_trial_care', '31a_Dissemination',
          '31b_Authorship', '31c_Sharing_Materials', '31d_Sharing_Data', '31e_Sharing_Code',
          '32_Informed_consent_materials', '33_Biological_specimens']
CONSORT = ['1a_Title_Randomized', '2_Abstract_structured', '3a_Registry_number', '3b_Protocol_access', '4_Funding',
           '7_Objectives', '8a_Design_Type', '8b_Design_Framework', '8c_Design_Centers', '8d_Design_Ratio', '9_Setting',
           '10a_Participants_inclusion', '10b_Center_interventionist_inclusion', '11a_Intervention_Description',
           '11b_Intervention_Modification', '11c_Intervention_Monitoring', '11d_Intervention_Concomitant',
           '12a_Outcomes_Definitions', '12b_Outcomes_Changes', '14a_Sample_size', '14b_Sample_Calculation',
           '16a_Randomization_Generation', '16b_Randomization_Type', '16c_Randomization_Block_size',
           '16d_Randomization_Strata', '16e_Allocation_Mechanism', '16f_Allocation_Concealment', '16g_Personnel_Sequence',
           '16h_Personnel_Enrollment', '17a_Masking_People_masked', '17b_Masking_Not_masked', '17c_Masking_Type',
           '17e_Masking_Similarity', '20a_Statistical_methods_Outcomes', '20b_Statistical_methods_Other_Analyses',
           '20c_Statistical_methods_Analysis_population', '21b_Interim_analyses', '21c_Stopping_guidelines', '34_Flow',
           '35a_Recruitment_dates', '35b_Followup_dates', '35c_Stopping', '36_Baseline_data', '37a_Analysis_Numbers',
           '38a_Outcome_results', '38b_Binary_results', '39_Ancillary_results', '40_Harms_results', '41_Generalizability',
           '25_Amendments']
def select_items(guideline, row):
    if guideline == 'CONSORT':
        return [i for i in row if i in CONSORT]
    elif guideline == 'SPIRIT':
        return [i for i in row if i in SPIRIT]
    else:
        return row

test = pd.read_csv('../data/test.csv', header=0)
test.Predictions = test.Predictions.apply(ast.literal_eval)
# test.SectionHeaders = test.SectionHeaders.apply(ast.literal_eval)
test.Predictions_item = test.Predictions_item.apply(ast.literal_eval)
test.Predictions_item = test.Predictions_item.map(lambda x: select_items('SPIRIT', x))

test.ChecklistItem = test.ChecklistItem.apply(ast.literal_eval)
test.ChecklistItem = test.ChecklistItem.map(lambda x: select_items('SPIRIT', x))



file = pd.read_csv('../data/sentence_level.csv', header=0)
file.ChecklistItem = file.ChecklistItem.apply(ast.literal_eval)
labels = set([j for i in file.ChecklistItem for j in i])
label_convert = MultiLabelBinarizer()
label_convert.fit([labels])
labels = label_convert.classes_


logit_result = label_convert.transform(test.Predictions_item.tolist()).tolist()
article_ids = test.PMCID.tolist()
sections = test.SectionHeaders.tolist()
target_result = label_convert.transform(test.ChecklistItem.tolist()).tolist()


# predict_items = []
# for predict in test.Predictions:
#     predict_items.append([labels[i] for i in range(len(predict)) if predict[i] == 1])
# test['Predictions_item'] = predict_items
# test.to_csv('../data/test.csv', index=False)


instance_report = calculate_multilabel_instance_metrics(logit_result, logit_result, target_result, labels,
                                                        sentence=True)
section_report = calculate_multilabel_section_metrics(logit_result, logit_result, target_result, labels, sections,
                                                      article_ids)
article_report = calculate_multilabel_article_metrics(logit_result, logit_result, target_result, labels, article_ids)

# Pretty print metrics
sec_keys = list(section_report.keys())
art_keys = list(article_report.keys())
print(f"{'Label' : <70}{'Instance' : ^20}{'Section' : ^20}{'Article' : >20}")
blank = 'N/A'
for i, inst_keys in enumerate(list(section_report.keys())):
    try:
        print(
            f"{inst_keys : <70}{instance_report[inst_keys]: ^20}{section_report[sec_keys[i]]: ^20}{article_report[art_keys[i]]: >20}")
    except:
        print(f"{inst_keys : <70}{blank: ^20}{section_report[sec_keys[i]]: ^20}{article_report[art_keys[i]]: >20}")
