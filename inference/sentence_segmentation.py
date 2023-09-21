import re


def merge_adjustment(i, sentences, segmentedsentence, current):
    segmentedsentence[-1]['sentence'] = segmentedsentence[-1]['sentence'] + current
    segmentedsentence[-1]['span'][1] = sentences[i]['span'][1]
    return segmentedsentence


def merge(sentences):
    """
    Merges sentences using regex based rules after the text has been initially split
    :param sentences: list of dictionaries containing 'sentence' (string) and 'span' (list)
    :return: list of dictionaries containing merged 'sentence' (string) and 'span' (list)
    """
    non_printable_pattern = re.compile(r"^(\s+)$")
    text_pattern = re.compile(r"[\w]")
    non_upper_pattern = re.compile(r"^[^A-Z]")
    conjunction_pattern = re.compile(r"^(for|and|nor|but|or|yet|so)")
    numbered_heading_pattern = re.compile(r"^\s*\d{1,2}\.(?:\d{1,2})?\.?\s*$")
    end_comma_pattern = re.compile(r", *(?!(\r?\n)+)$")
    end_pattern = re.compile(r"(?=(\r?\n)+)$")
    bacteria_pattern = re.compile(
        r"\b[A-Z]+(\.(\r?\n)+|\. +|\?(\r?\n)+|!(\r?\n)+|\? +|! +|(\r?\n)+)$")
    other_pattern = re.compile(
        r"\b([A-Z]|Figs*|et al|et al|i\.e|e\.g|vs|ca|min|sec|no|Dr|Inc|INC|[Cc]orp|Co|CO|Ltd|LTD|St|b\.i\.d|[Ee]tc|[Ii]\.[Vv])(\. +)$")
    other_num_pattern = re.compile(r"\d?\.\d+\s*$")
    midden_of_sentence_pattern = re.compile(
        r"\b(Figs*|[Ee]t [Aa]l|et al|[Ii]\.[Ee]|[Ee]\.[Gg]|vs|[Dd]r|[Dd]rs|[Pp]rof|[Nn]o|[Nn]r|[Mm]r|[Mm]s|[Ss]r|[Jj]r|[Ss]t|[Ss]q|b\.i\.d)(\. +)$")

    ori_n = len(sentences[0]['sentence'])
    sentences[0]['sentence'] = sentences[0]['sentence'].lstrip()
    new_n = len(sentences[0]['sentence'])
    sentences[0]['span'][0] += new_n - ori_n
    segmentedsentence = [sentences[0]]
    j = 0
    for i in range(1, len(sentences)):
        previous = segmentedsentence[j]['sentence'].lstrip()
        current = sentences[i]['sentence'].lstrip()
        if re.search(non_printable_pattern, current) or current.__len__() == 0:
            continue
        if re.search(midden_of_sentence_pattern, previous):
            segmentedsentence = merge_adjustment(i, sentences, segmentedsentence, current)
        elif re.search(numbered_heading_pattern, previous):
            segmentedsentence = merge_adjustment(i, sentences, segmentedsentence, current)
        elif re.search(conjunction_pattern, current):
            segmentedsentence = merge_adjustment(i, sentences, segmentedsentence, current)
        # if odd number of quotations or parentheses  is found in both sentences, merge
        elif (
                ((len(re.findall(r"[\"'“”]", previous)) % 2 != 0) and (len(re.findall(r"[\"'“”]", current)) % 2 != 0))
                or
                ((len(re.findall(r"[\(\)]", previous)) % 2 != 0) and (len(re.findall(r"[\(\)]", current)) % 2 != 0))
        ):
            segmentedsentence = merge_adjustment(i, sentences, segmentedsentence, current)
        elif (re.search(non_upper_pattern, current) and
              (re.search(bacteria_pattern, previous) or re.search(other_pattern, previous) or re.search(
                  other_num_pattern, previous))):
            segmentedsentence = merge_adjustment(i, sentences, segmentedsentence, current)
        elif not re.search(text_pattern, current) and not re.search(end_pattern, previous):
            segmentedsentence = merge_adjustment(i, sentences, segmentedsentence, current)
        elif re.search(end_comma_pattern, previous):
            segmentedsentence = merge_adjustment(i, sentences, segmentedsentence, current)
        else:
            j += 1
            segmentedsentence.append({'sentence': current, 'span': sentences[i]['span']})
    return segmentedsentence


def get_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def sort_check(section_header):
    section = []
    for i in section_header:
        n = int(len(section_header[i]) / 2)
        for j in range(n):
            section.append([i, section_header[i][2 * j], section_header[i][2 * j + 1]])
    section = sorted(section, key=lambda x: -x[1])
    remove_list = []
    for i in range(len(section)):
        if section[i][0] == '':
            for j in range(len(section)):
                if j != i:
                    com = get_overlap(section[i][1:], section[j][1:])
                    if com > 0:
                        remove_list.append(i)
    for i in sorted(set(remove_list), key=lambda x: -x):
        del section[i]
    return section


def segment(full_text, section_header=None):
    """
    Converts text documents into list of dictionaries containing 'sentence' (str) and 'span' (list)
    :param full_text: string object of the full text of a document
    :param section_header:
    :return: list of dictionaries containing merged 'sentence' (string) and 'span' (list)
    """
    pattern = re.compile(r"(.+?)" +
                         "(\. *(\r?\n)+|\? *(\r?\n)+|! *(\r?\n)+|\. +|\? +|! +|(\r?\n)+|" +
                         "\.[\"”]|" +
                         "[^0-9]\.[\"”]?[1-9][\[\]0-9,\-– ]*(?![.0-9]+)|" +
                         "[0-9]\.’?[1-9][\[\]0-9,\-– ]*(?=[A-Z][a-z])|$)"
                         )
    all_result = re.findall(pattern, full_text)
    sentences = []
    start = 0
    for res_tuple in all_result:
        text = ''.join(res_tuple[:2])
        while full_text[start] != text[0]:
            start += 1

        sentences.append({'sentence': text,
                          'span': [start, start + text.__len__() - 1]})
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
                if i['span'][0] >= span[0] - 1 and i['span'][1] <= span[1] + 1:
                    i['section'].append(sec[0])
    return sentences


if __name__ == '__main__':
    sentences = segment(
        'Samples were centrifuged at 1900 g, and the supernatant was collected and stored at 4°C. J774 cells were radiolabeled for 24\u2005hours in a medium containing 2 μCi of [3H]-cholesterol per microlitre.')
    print(sentences)
    p = re.compile(r"\b")
    print(re.search(p,'We \b assessed circulating EPC levels and EPC outgrowth number and function in CRS patients compared to healthy controls, and evaluated whether short-term (18 days) and long-term (52 weeks) EPO therapy improved EPC number and function in patients with CRS.\n      Methods'))