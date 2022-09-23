import re

NON_PRINTABLE_PATTERN = re.compile(r"^(\s+)$")
TEXT_PATTERN = re.compile(r"[\w]")
NON_UPPER_PATTERN = re.compile(r"^[^A-Z]")
END_COMMA_PATTERN = re.compile(r", *(?!(\r?\n)+)$")
END_PATTERN = re.compile(r"(?=(\r?\n)+)$")
BACTERIA_PATTERN = re.compile(
    r"\b[A-Z]+(\.(\r?\n)+|\. +|\?(\r?\n)+|!(\r?\n)+|\? +|! +|(\r?\n)+)$")
OTHER_PATTERN = re.compile(
    r"\b([A-Z]|Figs*|et al|et al|i\.e|e\.g|vs|ca|min|sec|no|Dr|Inc|INC|Co|CO|Ltd|LTD|St|b\.i\.d)(\. +)$")
MIDDEN_OF_SENTENCE_PATTERN = re.compile(r"\b([A-Z]|Figs*|et al|et al|i\.e|e\.g|vs|Dr|Drs|Prof|no|Ms|St|b\.i\.d)(\. +)$")


def merge(sentences):
    ori_n = len(sentences[0]['sentence'])
    sentences[0]['sentence'] = sentences[0]['sentence'].lstrip()
    new_n = len(sentences[0]['sentence'])
    sentences[0]['span'][0] += new_n - ori_n
    segmentedsentence = [sentences[0]]
    previous = sentences[0]['sentence'].lstrip()
    for i in range(1, len(sentences)):
        current = sentences[i]['sentence'].lstrip()
        if re.search(NON_PRINTABLE_PATTERN, current) or current.__len__() == 0:
            continue
        if re.search(MIDDEN_OF_SENTENCE_PATTERN, previous):
            segmentedsentence[-1]['sentence'] = segmentedsentence[-1]['sentence'] + current
            segmentedsentence[-1]['span'][1] = sentences[i]['span'][1]
        elif re.search(NON_UPPER_PATTERN, current) and (re.search(BACTERIA_PATTERN, previous) or re.search(OTHER_PATTERN, previous)):
            segmentedsentence[-1]['sentence'] = segmentedsentence[-1]['sentence'] + current
            segmentedsentence[-1]['span'][1] = sentences[i]['span'][1]
        elif not re.search(TEXT_PATTERN, current) and not re.search(END_PATTERN, previous):
            segmentedsentence[-1]['sentence'] = segmentedsentence[-1]['sentence'] + current
            segmentedsentence[-1]['span'][1] = sentences[i]['span'][1]
        elif re.search(END_COMMA_PATTERN, previous):
            segmentedsentence[-1]['sentence'] = segmentedsentence[-1]['sentence'] + current
            segmentedsentence[-1]['span'][1] = sentences[i]['span'][1]
        else:
            segmentedsentence.append({'sentence': current, 'span': sentences[i]['span']})
        previous = current
    return segmentedsentence

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def sort_check(section_header):
    section = []
    for i in section_header:
        n = int(len(section_header[i]) / 2)
        for j in range(n):
            section.append([i, section_header[i][2*j], section_header[i][2*j+1]])
    section = sorted(section, key = lambda x:-x[1])
    remove_list = []
    for i in range(len(section)):
        if section[i][0] == '':
            for j in range(len(section)):
                if j != i :
                    com = getOverlap(section[i][1:], section[j][1:])
                    if com > 0 :
                        remove_list.append(i)
    for i in sorted(set(remove_list), key=lambda x: -x):
        del section[i]
    return section


def segment(full_text, section_header=None):
    pattern = re.compile(r"(.+?)" +
							"(\. *(\r?\n)+|\? *(\r?\n)+|! *(\r?\n)+|\. +|\? +|! +|(\r?\n)+|" +
							"\.[\"”]|" +
							"[^0-9]\.[\"”]?[1-9][\[\]0-9,\-– ]*(?![.0-9]+)|" +
							"[0-9]\.’?[1-9][\[\]0-9,\-– ]*(?=[A-Z][a-z])|$)")
    all_result = re.findall(pattern, full_text)
    sentences = []
    start = 0
    for res_tuple in all_result:
        text= ''.join(res_tuple[:2])
        while full_text[start] != text[0]:
            start += 1

        sentences.append({'sentence': text,
                          'span':[start, start+text.__len__()-1]})
        start += text.__len__()

    sentences = merge(sentences)
    for i in sentences:
        ori = len(i['sentence'])
        i['sentence'] = i['sentence'].strip()
        after = len(i['sentence'])
        offset = ori - after
        if ori - after > 0:
            i['span'][1] -= offset

        if section_header:
            i['section'] = []
            for sec in section_header:
                span = sec[1:]
                if i['span'][0] >= span[0]-1 and i['span'][1] <= span[1]+1:
                    i['section'].append(sec[0])
    return sentences


if __name__ == '__main__':
    sentences =segment('Samples were centrifuged at 1900 g, and the supernatant was collected and stored at 4°C. J774 cells were radiolabeled for 24\u2005hours in a medium containing 2 μCi of [3H]-cholesterol per microlitre.')
    print(sentences)
    p = re.compile(r"\b")
    print(re.search(p, 'We \b assessed circulating EPC levels and EPC outgrowth number and function in CRS patients compared to healthy controls, and evaluated whether short-term (18 days) and long-term (52 weeks) EPO therapy improved EPC number and function in patients with CRS.\n      Methods'))