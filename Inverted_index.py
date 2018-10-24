import pymongo
import jieba
from functools import reduce
from string import punctuation as p
import re
jieba.load_userdict('../../Documents/wangsu_dict.txt')

mongoClient = pymongo.MongoClient('localhost', port=17088)
mongoClient.words.authenticate(name="cutword",
                               password="cutword@123",
                               mechanism="SCRAM-SHA-1",
                               source="admin")
mongoDB = mongoClient.words
mongoDBCollection = mongoDB['esim_service']

stop_words = []
with open('../../Documents/stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n')
        try:
            stop_words.append(line)
        except Exception as e:
            raise e


def word_split(text):
    '''
    Find the word position in a document.
    :param text: a document
    :return: [(position, word) ...]
    '''
    word_list = []
    stop_p = p + "~·！@#￥%……&*（）——=+-{}【】：；“”‘’《》，。？、|、"

    seg_list = [i for i in jieba.cut(text, cut_all=False) if (i != ' ' and i not in stop_p)]
    table = {}
    for i, c in enumerate(seg_list):
        if c in table:
            table[c] += 1
        else:
            table.setdefault(c, 1)

        value = ([i for i, word in enumerate(seg_list) if word == c][table[c] - 1], c.lower())
        word_list.append(value)

    return word_list, table


def word_cleanup(words):
    cleaned_words = [(pos, word) for (pos, word) in words if word not in stop_words]
    return cleaned_words


def word_index(text):
    words, table = word_split(text)
    words = word_cleanup(words)
    return words, table


def inverted_index(text):
    '''
    Get position of each word in a document.
    :return: {'word1':[position1, position2 ...], 'word2':...} & table(word_count)
    '''
    inverted = {}
    word_list, table = word_index(text)

    for position, word in word_list:
        locations = inverted.setdefault(word, [])
        locations.append(position)

    return inverted, table


def inverted_index_add(dict_index, doc_id, doc_index):
    '''
    Get position of each word in all documents.
    :return: {'word1': {'doc1': [pos1, pos2, ...], 'doc2': ...}, 'word2': ...}
    '''
    for word, locations in doc_index.items():
        indices = dict_index.setdefault(word, {})
        indices[doc_id] = locations


def write_down_the_sequence(seq_list):
    chinese_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    flag = True

    sent = seq_list[0]
    if chinese_pattern.search(sent):
        flag = False

    for word in seq_list[1:]:
        if chinese_pattern.search(word):
            sent += word
            flag = False
        else:
            if flag:
                sent = sent + " " + word
            else:
                sent += word
                flag = True

    return sent


def neighbor_search(result_set, pos_list, range, words):
    '''
    :param pos_list: [[pos1, pos2, ... for word1], [pos1, ... for word2], ...]
    '''
    left_buf, left_indices = [set(map(lambda x: x + range, pos_list[0]))], [list(words[0])]
    for idx in range(1, len(pos_list)):
        right = set(pos_list[idx])
        result = [left_element & right for left_element in left_buf]
        flag = reduce(lambda x, y: x and y, map(lambda x: len(x) == 0, result))
        if not flag:
            display_index = [result.index(i) for i in result if len(i) != 0]
            left_indices = [i.append(words[idx]) for i in left_indices]
            for item in display_index:
                result_set.append((write_down_the_sequence(left_indices[item]), len(result[item])))

            left_buf = [set(map(lambda x: x + range, suc)) for suc in result]
        else:
            left_indices.append(list(words[idx]))
            left_buf.append(right)
            left_buf = [set(map(lambda x: x + range, suc)) for suc in left_buf]


def search(dict_index, query, counter_table, tolerate=3, search_for_all=True):
    '''
    :param dict_index: {'word1': {'doc1': [pos1, pos2, ...], 'doc2': ...}, 'word2': ...}
    :param counter_table: {'doc1': {'word1': num1, 'word2': num2, ...}, 'doc2' : ...}
    '''
    word_list, _ = word_index(query)
    words = [word for _, word in word_list if word in dict_index]
    doc_set = [set(dict_index[word].keys()) for word in words]
    query_result = {}

    # 必须包含所有的word的doc才会被筛选出来
    if search_for_all:
        doc_set = reduce(lambda x, y: x & y, doc_set) if doc_set else []
        # doc_set : [doc_id1, doc_id2, ...] for doc contain all words

        if doc_set:
            for doc in doc_set:
                result_set = query_result.setdefault(doc, [])
                # 先把单个词的情况添加进去
                for word in words:
                    result_set.append((word, counter_table[doc][word]))

                pos_list = [[i for i in dict_index[word][doc]] for word in words]

                for i in range(1, tolerate + 1):
                    neighbor_search(result_set, pos_list, i, words)

    # 只要有关键字包含其中，就会被检索出来，无论多少
    else:
        # doc_set : [[doc_id for word1], [doc_id for word2], ...]
        total_set = reduce(lambda x, y: x | y, doc_set)
        # total_set : [doc_id1, doc_id2, ...]
        for doc in total_set:
            result_set = query_result.setdefault(doc, [])
            words = [word for idx, word in enumerate(words) if doc in list(doc_set)[idx]]

            for word in words:
                result_set.append((word, counter_table[doc][word]))

            pos_list = [[i for i in dict_index[word][doc]] for word in words]

            for i in range(1, tolerate + 1):
                neighbor_search(result_set, pos_list, i, words)

    return query_result


def retrieval_ans(queries):
    '''
    Find the most fitting doc. from knowledge-base using inverted retrieval.
    '''
    # 导出知识库所有内容
    inverted, knowledge_list, zone_types = {}, {}, []
    for item in mongoDBCollection.find():
        content = item['question'] + item['answer']
        knowledge_list.setdefault(item['question_id'], content)
        if item['zone'] not in zone_types:
            zone_types.append(item['zone'])

    # 建立倒排索引表
    tables = {}
    for question_id, content in knowledge_list.items():
        doc_index, table = inverted_index(content)
        tables[question_id] = table
        inverted_index_add(inverted, question_id, doc_index)

    # 对检索知识库的结果进行排序
    for query in queries:
        result = search(inverted, query, tables)
        print('Search for {} ...'.format(query))
        for key, value in result.items():
            print('Result is {}: {}'.format(key, value))

    print('Done!')


if __name__ == '__main__':
    retrieval_ans(["portal账号如何注册？"])
