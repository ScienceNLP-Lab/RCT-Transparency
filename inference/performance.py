import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import ast

correct = pd.read_csv('all_CONSORT_manual_data.csv', header=0)
correct['CONSORT_Item'] = correct['CONSORT_Item'].apply(ast.literal_eval)
print(correct['CONSORT_Item'].tolist())
predict = pd.read_csv('prediction_covid.csv', header=0)
predict['label'] = predict['label'].apply(ast.literal_eval)
print(predict['label'].tolist())

label_convert = MultiLabelBinarizer()
label_convert.fit([list(set([j for i in correct['CONSORT_Item'] for j in i if j!='0']))])


print(classification_report(label_convert.transform(correct['CONSORT_Item']), label_convert.transform(predict['label'])))