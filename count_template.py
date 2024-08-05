import pandas as pd

def map_labels(row):
    if row['Label'] == '-':
        return '-'
    else:
        return 'anomaly'

if __name__ == "__main__":
    log_name = 'Thunderbird.log'
    structured_log_path = f'dataset/thunderbird/{log_name}_structured.csv'
    template_file_path = f'dataset/thunderbird/{log_name}_templates.csv'


    template_df = pd.read_csv(template_file_path)
    log_df = pd.read_csv(structured_log_path)
    selected_log_df = log_df.loc[:, ['Label', 'EventId', 'EventTemplate']]
    # selected_log_df['Label'] = selected_log_df.apply(lambda row: map_labels(row), axis=1)
    selected_log_df = selected_log_df.drop_duplicates(subset=['EventId'])
    indexed_log_df = selected_log_df.set_index('EventId')
    dict_log = indexed_log_df.to_dict(orient='index')

    normal_num = 0
    abnormal_num = 0
    for index, template in template_df.iterrows():
        eventId = template['EventId']
        template_dict = dict_log[eventId]
        if template_dict['Label'].strip() == '-':
            print(template_dict)
            normal_num += 1
        else:
            print(template_dict)
            abnormal_num += 1

    print('abnormal_num', abnormal_num)
    print('normal_num', normal_num)
