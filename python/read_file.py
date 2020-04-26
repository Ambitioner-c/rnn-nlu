import demjson
import re

# 标点符号
remove_chars = '[·’!"#$%&\'*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'

with open('./q_a_data_5_300.txt', 'r') as f_r:
    total = 0
    # 车型对比:1
    model = 0
    # 汽车属性:2
    attribute = 0
    # 汽车推荐:3
    recommend = 0
    # 用户观点:4
    viewpoint = 0
    # 其他:5
    other = 0
    for j in range(5 * 300):
        txt = f_r.readline()
        txt_json = demjson.decode(txt)

        txt = txt_json['original_Q']
        txt = re.sub(remove_chars, "", txt)

        if txt_json['intention'] == 1:
            model += 1
            if 240 < model <= 300:
                with open('../data/ATIS_samples/test/test.label', 'a') as f_a:
                    f_a.writelines('1' + '\n')

        if txt_json['intention'] == 2:
            attribute += 1
            if 240 < attribute <= 300:
                with open('../data/ATIS_samples/test/test.label', 'a') as f_a:
                    f_a.writelines('2' + '\n')

        if txt_json['intention'] == 3:
            recommend += 1
            if 240 < recommend <= 300:
                with open('../data/ATIS_samples/test/test.label', 'a') as f_a:
                    f_a.writelines('3' + '\n')

        if txt_json['intention'] == 4:
            viewpoint += 1
            if 240 < viewpoint <= 300:
                with open('../data/ATIS_samples/test/test.label', 'a') as f_a:
                    f_a.writelines('4' + '\n')

        if txt_json['intention'] == 5:
            other += 1
            if 240 < other <= 300:
                with open('../data/ATIS_samples/test/test.label', 'a') as f_a:
                    f_a.writelines('5' + '\n')
