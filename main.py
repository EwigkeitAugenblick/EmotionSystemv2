import main_st2
import pandas as pd
number=0
list_vector = pd.read_csv('示例视频列表-向量.csv', encoding='utf-8', sep=';')
vector = list_vector[list_vector.columns[:48]].values[int(number)].reshape(1, 48)
feature_list, pred, video_number = main_st2.reason_confer(vector)
if pred[0] == 0:
    emotion = '不会'
else:
    emotion = '会'
result = "根据与此视频{}，{}最相关的此三个视频的{}特征判断，此视频{}产生极端性群体情绪.".format(
                                feature_list[0], feature_list[1], feature_list[2], emotion)
