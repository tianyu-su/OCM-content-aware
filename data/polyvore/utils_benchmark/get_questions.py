import json
import numpy as np
import os.path as osp


def get_questions(data_path):
    json_file = osp.join(data_path, "test.json")
    questions_file = osp.join(data_path, "fill_in_blank_test.json")
    with open(json_file) as f:
        json_data = json.load(f)

    with open(questions_file) as f:
        questions_data = json.load(f)

    print('There are {} questions.'.format(len(questions_data)))
    print('There are {} test outfits.'.format(len(json_data)))

    # map ids in the form 'outfit_index' to 'imgID', because in 'fill_in_blank_test.json'
    # the ids are in format outfit_index but I use the other ID (so that the same item in two outfits has the same id)
    map_ids = {}
    for outfit in json_data:
        for item in outfit['items']:
            outfit_id = '{}_{}'.format(outfit['set_id'], str(item['index']))
            map_ids[outfit_id] = item['item_id']

    save_data = []
    counts = 0
    count_all_valid = 0
    for ques in questions_data:
        q = []
        for q_id in ques['question']:
            outfit_id = q_id.split('_')[0]
            q_id = map_ids[q_id]
            q.append(q_id)
        a = []
        positions = []
        # add right answer, because maybe not right answer at first position
        right_id = f"{outfit_id}_{ques['blank_position']}"
        a.append(map_ids[right_id])
        positions.append(ques['blank_position'])
        for a_id in ques['answers']:
            if a_id == right_id:
                continue
            else:
                if a_id.split('_')[0] == outfit_id:
                    pass  # this is true for a few edge queries
            pos = int(a_id.split('_')[-1])  # get the posittion of this item within the outfit
            a_id = map_ids[a_id]
            a.append(a_id)
            positions.append(pos)

        # count how many questions have only one possible choice of the correct category
        choices = sum([p == ques['blank_position'] for p in positions])
        counts += choices == 1
        count_all_valid += choices == 4
        save_data.append([q, a, positions, ques['blank_position']])

    # save_data is a list of questions
    # each question is a list that contains:
    #    - list of outfit IDs (len N)
    #    - list of possible answers (len 4)
    #    - list of possible answers positions
    #    - desired position
    return save_data
