import json
import os.path as osp


def get_compats(data_path):
    #################
    #### MAP IDS ####
    #################

    json_file = osp.join(data_path, "test.json")

    with open(json_file) as f:
        json_data = json.load(f)

    # map ids in the form 'outfit_index' to 'imgID', because in 'fill_in_blank_test.json'
    # the ids are in format outfit_index but I use the other ID (so that the same item in two outfits has the same id)
    map_ids = {}
    for outfit in json_data:
        for item in outfit['items']:
            outfit_id = '{}_{}'.format(outfit['set_id'], str(item['index']))
            map_ids[outfit_id] = item['item_id']

    ################################
    #### PROCESS COMPAT OUTFITS ####
    ################################

    compat_file = osp.join(data_path, "compatibility_test.txt")

    outfits = []
    n_comps = 0
    with open(compat_file) as f:
        for line in f:
            cols = line.rstrip().split(' ')
            compat_score = float(cols[0])
            assert compat_score in [1, 0]
            # map their ids to my img ids
            items = []
            for it in cols[1:]:
                if it:
                    items.append(map_ids[it])

            n_comps += 1
            outfits.append((items, compat_score))

    print('There are {} outfits to test compatibility'.format(n_comps))
    print('There are {} test outfits.'.format(len(json_data)))

    # returns 2 lists:
    # - outfits: len N, contains lists of outfits
    # - labels: len N, contains the labels corresponding to the outfits
    return outfits
