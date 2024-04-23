from st_on_hover_tabs import on_hover_tabs
import streamlit as st
from streamlit import runtime
import sys
from streamlit.web import cli as stcli
import random
import time
import base64
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import re
import json
import jieba
import xgboost as xgb
import all_sort
import os



def reason_confer(vector_list):
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model("xgb_model.model")
    y_pred = loaded_model.predict(vector_list)
    feature_importances = loaded_model.feature_importances_
    author = [0, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39]
    rec = [4, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43]
    key = [8, 9, 10, 11, 20, 21, 21, 23, 32, 33, 34, 35, 44, 45, 46, 47]

    title = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]
    tag = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    intro = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46]
    cover = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]

    cluster = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    fans = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    play = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    hot = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    feature_list = []
    prevideo_list = []
    k = np.argmax(feature_importances)
    if k in author:
        feature_list.append('上下文相关')
    if k in rec:
        feature_list.append('大数据相关')
    if k in key:
        feature_list.append('内容相关')
    if k in title:
        feature_list.append('标题')
    if k in tag:
        feature_list.append('标签')
    if k in intro:
        feature_list.append('简介')
    if k in cover:
        feature_list.append('封面')
    if k in cluster:
        feature_list.append('集群密度')
    if k in fans:
        feature_list.append('粉丝数')
    if k in play:
        feature_list.append('播放量')
    if k in hot:
        feature_list.append('热度')
    if '上下文相关' in feature_list:
        if '标题' in feature_list:
            prevideo_list = [0, 1, 2]
        elif '标签' in feature_list:
            prevideo_list = [3, 4, 5]
        elif '简介' in feature_list:
            prevideo_list = [6, 7, 8]
        elif '封面' in feature_list:
            prevideo_list = [9, 10, 11]
    if '大数据相关' in feature_list:
        if '标题' in feature_list:
            prevideo_list = [12, 13, 14]
        elif '标签' in feature_list:
            prevideo_list = [15, 16, 17]
        elif '简介' in feature_list:
            prevideo_list = [18, 19, 20]
        elif '封面' in feature_list:
            prevideo_list = [21, 22, 23]
    if '内容相关' in feature_list:
        if '标题' in feature_list:
            prevideo_list = [24, 25, 26]
        elif '标签' in feature_list:
            prevideo_list = [27, 28, 29]
        elif '简介' in feature_list:
            prevideo_list = [30, 31, 32]
        elif '封面' in feature_list:
            prevideo_list = [33, 34, 35]


    return feature_list, y_pred, prevideo_list


headers = {
    'authority': 'api.bilibili.com',
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    # 需定期更换cookie，否则location爬不到
    'cookie': "i-wanna-go-back=-1; header_theme_version=CLOSE; nostalgia_conf=-1; CURRENT_PID=48e7c380-d774-11ed-b043-a76171efa2f7; DedeUserID=3493136064056111; DedeUserID__ckMd5=0d65c5fa36863cf3; enable_web_push=DISABLE; hit-dyn-v2=1; CURRENT_BLACKGAP=0; fingerprint=8edb046d8598423e594ff305a0f49ace; buvid_fp_plain=undefined; CURRENT_FNVAL=4048; PVID=1; buvid3=2E4C2934-913D-2351-4300-3AEE113D4B6E07674infoc; b_nut=1705226307; b_ut=7; _uuid=9C2A5519-5E41-10CC1-99CE-BB32EF9E735B37279infoc; buvid_fp=42e68d1839873d3b0a85e26bc2fd2619; buvid4=EEA57664-EEA9-B54E-B6AF-3F09F8E51E1208443-024011409-HJwPMewZWLS8kLGKMCXTQA%3D%3D; rpdid=|(k|YumYuluu0J'u~|Yk)|)Y~; home_feed_column=4; CURRENT_QUALITY=0; bp_video_offset_3493136064056111=908667290013139012; browser_resolution=1327-756; FEED_LIVE_VERSION=V_WATCHLATER_PIP_WINDOW3; b_lsid=49482795_18E8D87D57B; bsource=search_bing; SESSDATA=094fd50b%2C1727323475%2C61487%2A31CjDpL5Za4XTa-9OrvC3naXRPD2ZyX-9bnJPyF4EIjc3YtdUdUsfdS6luEyQNGAMsbTMSVkd2dDNHZDdIaUJ6bkh0RTJIV0UyZUlFSjVXbHlsR1ZlOGFfYWpoeFJWUTd1SzhJV2c5WG5ScmFpU2J1M3VqN1d6Wk4wMEdTbFhSVDBqZzUxUzlJTHNnIIEC; bili_jct=3a03a34244cb1923cf32fb1f863a81c5; sid=78gn4u1u; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTIwMzA2NzYsImlhdCI6MTcxMTc3MTQxNiwicGx0IjotMX0.n5HBM6pi3-OwuOVBIyd94eiD6eUwnOqhGvuhuEPlBks; bili_ticket_expires=1712030616",
    'origin': 'https://www.bilibili.com',
    'referer': 'https://www.bilibili.com/video/BV1FG4y1Z7po/?spm_id_from=333.337.search-card.all.click&vd_source=69a50ad969074af9e79ad13b34b1a548',
    'sec-ch-ua': '"Chromium";v="106", "Microsoft Edge";v="106", "Not;A=Brand";v="99"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.47'
}


def load_lottie_json(path):
    with open(path, "r") as f:
        lottie_json = f.read()
    return lottie_json


def initial():
    global keyword
    keyword = None
    global website
    website = None
    global key
    key = None
    global web
    web = None
    global uploaded_file1
    uploaded_file1 = None
    global uploaded_file2
    uploaded_file2 = None


def progress_bar(max_time):
    start_time = time.time()
    progress_placeholder = st.empty()  # 创建一个占位符
    for i in range(1, 101):
        num_progress = i
        progress_placeholder.markdown(f''' 
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Progress_bar</title>
                <link rel="stylesheet" href="progress_bar.css">
            </head>
            <body>
                <div class="container">
                    <section>
                        <article>
                            <!-- <input type="radio" name="switch-color" id="red" checked>
                            <input type="radio" name="switch-color" id="cyan">
                            <input type="radio" name="switch-color" id="lime"> -->
                            <div class="chart">
                                <div class="bar bar-{num_progress} cyan">
                                    <div class="face top">
                                        <div class="growing-bar"><p style="float: right; margin-right:10px"><strong>{num_progress}</strong>%</p></div>
                                    </div>
                                    <div class="face side-0">
                                        <div class="growing-bar"></div>
                                    </div>
                                    <div class="face floor">
                                        <div class="growing-bar"></div>
                                    </div>
                                    <div class="face side-a"></div>
                                    <div class="face side-b"></div>
                                    <div class="face side-1">
                                        <div class="growing-bar"></div>
                                    </div>
                                </div>
                            </div>
                        </article>
                    </section>
                </div> 
            </body>
            </html>''', unsafe_allow_html=True)
        elapsed_time = time.time() - start_time
        if elapsed_time > max_time:
            break
        if i < 100:
            sleep_time = random.uniform(0.01, (max_time - elapsed_time) / (100 - i))
            time.sleep(sleep_time)
            progress_placeholder.empty()


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):  # 设置背景图
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/ipg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


def upload_file_to_0x0(file_path):
    """上传文件到0x0并返回URL"""
    with open(file_path, 'rb') as f:
        response = requests.post('https://0x0.st', files={'file': f})
    if response.status_code == 200:
        return response.text.strip()
    else:
        return None





def get_middle_part(file_name):
    # 去除文件扩展名
    file_name = file_name.split('.')[0]
    # 按照 "-" 分割文件名
    parts = file_name.split('-')

    if len(parts) > 1:
        middle_part = parts[1]
    else:
        middle_part = parts[0]

    return middle_part


def change_date(times):
    time2 = []
    for time1 in times:
        time1 = str(time1)[:10]
        time2.append(time1)
    return time2


def find_imppost_data(file_name):
    file_name = re.sub(r'\s?\([^)]*\)', '', file_name)
    path1 = get_middle_part(file_name)
    if 'xlsx' in file_name:
        data = pd.read_excel(path1 + "/" + file_name)
    else:
        data = pd.read_csv(path1 + "/" + file_name, encoding='utf-8', sep=';')
    data['评论时间'] = change_date(data['评论时间'].values)
    df = data.sort_values(by="评论时间", ascending=True)
    df['评论时间'] = pd.to_datetime(df['评论时间'])
    df['日期'] = df['评论时间'].dt.strftime('%Y-%m-%d')
    # print(df['日期'].to_string())
    df_deduplicated = df.drop_duplicates(subset=['发布者', '文本'], keep='first')
    df_sorted = df_deduplicated.sort_values(['日期', '点赞数'], ascending=[True, False])
    posts_dict_poster = pd.Series(df_sorted[['发布者']].values.tolist(), index=df_sorted['日期']).to_dict()
    posts_dict = pd.Series(df_sorted[['文本', '发布者']].values.tolist(), index=df_sorted['日期']).to_dict()
    # print(posts_dict)
    return (posts_dict, posts_dict_poster)


def match_url(dict_, csv_file):
    csv_file = re.sub(r'\s?\([^)]*\)', '', csv_file)
    url_data = pd.read_csv(csv_file, encoding='utf-8', sep=';')
    result_dict = {}
    for date, publisher in dict_.items():
        date_str = datetime.strptime(date, '%Y-%m-%d').strftime('%m月%d日')
        matched_row = url_data[(url_data['发布时间'].str.startswith(date_str)) & (url_data['发布者'] == publisher[0])]
        if not matched_row.empty:
            result_dict[date] = matched_row['博客url链接'].values[0]
        else:
            # 如果当天没有找到匹配的数据，尝试在前几天找
            for i in range(1, 8):  # 尝试在前7天找
                prev_date_str = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=i)).strftime('%m月%d日')
                matched_row_prev = url_data[
                    (url_data['发布时间'].str.startswith(prev_date_str)) & (url_data['发布者'] == publisher[0])]
                if not matched_row_prev.empty:
                    result_dict[date] = matched_row_prev['博客url链接'].values[0]
                    break
            else:
                result_dict[date] = None
    # print(result_dict)
    return result_dict


def videoshow(i, related_video):
    pic = related_video['封面图片'].values[i]
    title = related_video['标题'].values[i]
    tag = related_video['标签'].values[i]
    intro = related_video['简介'].values[i]

    picture = requests.get(pic, stream=True)
    with open('第' + str(i) + '条相关视频封面.png', "wb") as png:
        for chunk in picture.iter_content(chunk_size=1024):
            if chunk:
                png.write(chunk)
    st.write(f'<p style="font-size:20px;font-weight:bold;">{title}</p >',
             unsafe_allow_html=True)
    st.image('第' + str(i) + '条相关视频封面.png', caption='视频封面', width=280)
    st.write('<p style="font-size:18px;font-weight:bold;">标签：</p >',
             unsafe_allow_html=True)
    st.write(tag)
    st.write('<p style="font-size:18px;font-weight:bold;">简介：</p >',
             unsafe_allow_html=True)
    st.write(intro)


def main():
    initial()
    st.session_state.style = "标准情绪值"

    # st.session_state.file_in=" "
    # st.session_state.file_out=" "
    st.set_page_config(layout="wide")
    set_background("bg.png")  ##更改点（背景图）
    col, col_title, col = st.columns([3, 4, 1])
    with col_title:
        st.header("极端群体情绪预测系统")
    st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        tabs = on_hover_tabs(
            tabName=['Video Search', 'Emotion Prediction'],
            iconName=[ 'money', 'economy'], default_choice=0)

    if tabs == 'Emotion Prediction':
        st.session_state.file_in = " "
        st.subheader('极端群体情绪预测')
        model = st.selectbox('请选择预测模式:', (' ', '实时模式', '演示模式'))
        if model == '实时模式':
            uploaded_file3 = st.file_uploader("上传相关视频文件")
            if uploaded_file3:
                data2 = pd.read_csv(uploaded_file3, encoding='utf-8', engine="python")
                video_urls = data2['urls'].values
                st.text('正在构建预测向量，请等待5到10分钟...')
                vector_list = all_sort.vector_get(video_urls)
                st.text('正在预测...')
                vector_list = np.array(vector_list).reshape(1, 48)
                feature_list, pred, video_number = reason_confer(vector_list)
                if pred[0] == 0:
                    emotion = '不会'
                else:
                    emotion = '会'
                result = "根据与此视频{}，{}最相关的此三个视频的{}特征判断，此视频{}产生极端性群体情绪.".format(
                    feature_list[0], feature_list[1], feature_list[2], emotion)

                if result:
                    col_ex1, col_ex2, col_ex3 = st.columns([1, 1, 1])
                    # with open('progress_bar.css', 'r', ) as f:
                    # st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
                    k = st.markdown(f"<h1 style='text-align: center;'>此视频<strong>{emotion}产生极端群体情绪</h1>",
                                    unsafe_allow_html=True)
                    st.success(result)
                    if k:
                        with col_ex1:
                            with st.expander("依据视频1", True):
                                videoshow(video_number[0], data2)
                        with col_ex2:
                            with st.expander("依据视频2", True):
                                videoshow(video_number[1], data2)
                        with col_ex3:
                            with st.expander("依据视频3", True):
                                videoshow(video_number[2], data2)

        if model == '演示模式':
            list = pd.read_csv('示例视频列表.csv', encoding='utf-8')
            list_new = list[['标题', '关键词-自定义', '标签', '简介']]
            col1, col2 = st.columns([1, 1])
            with col1:
                with st.spinner('Wait about 3 seconds'):
                    with st.expander("本地数据库", True):
                        number = st.text_input('请输入演示视频编号:')
                        if number:
                            list_vector = pd.read_csv('示例视频列表-向量.csv', encoding='utf-8', sep=';')
                            vector = list_vector[list_vector.columns[:48]].values[int(number)].reshape(1, 48)
                            feature_list, pred, video_number = reason_confer(vector)
                            if pred[0] == 0:
                                emotion = '不会'
                            else:
                                emotion = '会'
                            result = "根据与此视频{}，{}最相关的此三个视频的{}特征判断，此视频{}产生极端性群体情绪.".format(
                                feature_list[0], feature_list[1], feature_list[2], emotion)
                            pic = list['封面图片'].values[int(number)]
                            title = list['标题'].values[int(number)]
                            tag = list['标签'].values[int(number)]
                            intro = list['简介'].values[int(number)]
                            url = list['urls'].values[int(number)]

                            picture = requests.get(pic, stream=True)
                            with open('相关视频' + number + '封面.png', "wb") as png:
                                for chunk in picture.iter_content(chunk_size=1024):
                                    if chunk:
                                        png.write(chunk)
                            st.write(f'<a href="{url}"><p style="font-size:20px;font-weight:bold;">{title}</p ></a >',
                                     unsafe_allow_html=True)
                            col_ex1, col_ex2 = st.columns([1, 1])
                            with col_ex1:
                                st.image('相关视频' + number + '封面.png', caption='视频封面', width=380)
                            with col_ex2:
                                st.write('<p style="font-size:18px;font-weight:bold;">标签：</p >',
                                         unsafe_allow_html=True)
                                st.write(tag)
                                st.write('<p style="font-size:18px;font-weight:bold;">简介：</p >',
                                         unsafe_allow_html=True)
                                st.write(intro)

            with col2:
                with st.expander('可选择视频列表', True):
                    st.write(list_new)
            if number:
                project_path = os.getcwd()
                file = project_path+'\相关视频\第' + number + '条链接相关视频.csv'
                related_video = pd.read_csv(file, encoding='utf-8')
                col_ex1, col_ex2, col_ex3 = st.columns([1, 1, 1])
                # with open('progress_bar.css', 'r', ) as f:
                # st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
                k = st.markdown(f"<h1 style='text-align: center;'>此视频<strong>{emotion}产生极端群体情绪</h1>",
                                unsafe_allow_html=True)
                st.success(result)
                if k:
                    with col_ex1:
                        with st.expander("依据视频1", True):
                            videoshow(video_number[0], related_video)
                    with col_ex2:
                        with st.expander("依据视频2", True):
                            videoshow(video_number[1], related_video)
                    with col_ex3:
                        with st.expander("依据视频3", True):
                            videoshow(video_number[2], related_video)

    if tabs == 'Video Search':
        st.session_state.file_in = " "
        st.subheader('相关视频搜索')
        model = st.selectbox('请选择预测模式:', (' ', '实时模式', '演示模式'))
        if model == '实时模式':
            keyword = st.text_input('请输入视频关键词，如果不指定关键词，请输入0:')
            url = st.text_input('请输入视频地址:')
            if url:
                bv_num = re.findall(r'BV\w+', url)
                BV = ''.join(bv_num)
                if BV:
                    time.sleep(0.1)
                    video_api = 'https://api.bilibili.com/x/web-interface/view'
                    tag_api = 'https://api.bilibili.com/x/tag/archive/tags'
                    params = {'bvid': BV}
                    response = requests.get(video_api, params=params, headers=headers)
                    response.encoding = 'utf-8-sig'
                    videos = json.loads(response.text)
                    video = videos['data']
                    pic = video['pic']
                    picture = requests.get(pic, stream=True)
                    with open('预测视频封面图片.png', "wb") as png:
                        for chunk in picture.iter_content(chunk_size=1024):
                            if chunk:
                                png.write(chunk)
                    title = video['title']
                    try:
                        intro1 = video['desc_v2'][0]['raw_text']
                        intro = intro1.replace('\n', ' ')
                    except:
                        intro = '-'

                    tag = ''
                    response2 = requests.get(tag_api, params=params, headers=headers)
                    response2.encoding = 'utf-8-sig'
                    tags = json.loads(response2.text)['data']
                    for k in tags:
                        tag = tag + k['tag_name'] + ' '
                    if keyword =='0':
                        keyword=title
                    keyword = keyword.replace('习近平', "")


                    rec_bvid = []
                    url_related = 'https://api.bilibili.com/x/web-interface/archive/related'
                    params = {
                        'bvid': BV
                    }
                    response = requests.get(url_related, params=params, headers=headers)
                    response.encoding = 'utf-8-sig'
                    rec_video_urls = json.loads(response.text)['data']
                    for rec_video_url in rec_video_urls:
                        rec_bvid.append(rec_video_url['bvid'])

                    st.text('请在弹出页面后全屏网页并手动登录B站（20秒时间）')
                    publisher_bvid = all_sort.publisher_bv(BV, params)
                    st.text('正在搜索，请等待5到10分钟...')
                    keyword_bvid = all_sort.keyword_bv2(keyword, BV)

                    st.text('正在检索上下文相关视频...')
                    publisher_data = all_sort.data_crawler(publisher_bvid)
                    st.text('正在检索大数据相关视频...')
                    rec_data = all_sort.data_crawler(rec_bvid)
                    st.text('正在检索内容相关视频...')
                    keyword_data = all_sort.data_crawler(keyword_bvid)

                    st.text('正在排序上下文相关视频...')
                    video_publisher = all_sort.video_select(publisher_data, title, tag, intro, pic)
                    st.text('正在排序大数据相关视频...')
                    video_keyword = all_sort.video_select(rec_data, title, tag, intro, pic)
                    st.text('正在排序内容相关视频...')
                    video_rec = all_sort.video_select(keyword_data, title, tag, intro, pic)
                    video = pd.concat([video_publisher, video_keyword, video_rec])
                    video.to_csv(BV+'相关视频信息.csv')
                    video_data=video.to_csv(index=False)


                    progress_text = "正在搜索本视频的相关视频..."
                    my_bar = st.progress(0, text=progress_text)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1, text=progress_text)
                    time.sleep(1)
                    my_bar.empty()
                    st.success('搜索完成！')
                    st.success('本视频相关视频已保存为csv文件')
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        with st.expander("预测视频信息", True):
                            st.write(f'<a href="{url}"><p style="font-size:28px;font-weight:bold;">{title}</p ></a >',
                                     unsafe_allow_html=True)
                            col_ex1, col_ex2 = st.columns([1, 1])
                            with col_ex1:
                                st.image('预测视频封面图片.png', caption='视频封面', width=400)
                            with col_ex2:
                                st.write('<p style="font-size:18px;font-weight:bold;">标签：</p >',
                                         unsafe_allow_html=True)
                                st.write(tag)
                                st.write('<p style="font-size:18px;font-weight:bold;">简介：</p >',
                                         unsafe_allow_html=True)
                                st.write(intro)
                        with st.expander("相关视频信息", True):
                            data1 = pd.read_csv(BV+'相关视频信息.csv', encoding='utf-8', engine="python")
                            st.write(data1)
                    st.download_button("下载数据", video_data, file_name=BV+'相关视频信息.csv', mime='text/csv')

            # bv_num = re.findall(r'BV\w+', url)
            # BV = ''.join(bv_num)
            # if BV:
            # time.sleep(0.1)
            # video_api = 'https://api.bilibili.com/x/web-interface/view'
            # tag_api = 'https://api.bilibili.com/x/tag/archive/tags'
            # params = {
            # 'bvid': BV
            # }
            # response = requests.get(video_api, params=params, headers=headers)
            # response.encoding = 'utf-8-sig'
            # videos = json.loads(response.text)
            # video = videos['data']
            # pic = video['pic']
            # picture = requests.get(pic, stream=True)
            # with open('封面图片.png', "wb") as png:
            # for chunk in picture.iter_content(chunk_size=1024):
            # if chunk:
            # png.write(chunk)

            # title = video['title']
            # try:
            # intro1 = video['desc_v2'][0]['raw_text']
            # intro = intro1.replace('\n', ' ')
            # except:
            # intro = '-'

            # tag = ''
            # response2 = requests.get(tag_api, params=params, headers=headers)
            # response2.encoding = 'utf-8-sig'
            # tags = json.loads(response2.text)['data']
            # for k in tags:
            # tag = tag + k['tag_name'] + ' '

            # if keyword == '0':
            # keyword = title

            # rec_bvid = []
            # match = re.findall(r'BV\w+', url)
            # BV = ''.join(match)
            # url_related = 'https://api.bilibili.com/x/web-interface/archive/related'
            # params = {
            # 'bvid': BV
            # }
            # response = requests.get(url_related, params=params, headers=headers)
            # response.encoding = 'utf-8-sig'
            # rec_video_urls = json.loads(response.text)['data']
            # for rec_video_url in rec_video_urls:
            # rec_bvid.append(rec_video_url['bvid'])

            # keyword_bvid = all_sort.keyword_bv(keyword, BV)
            # publisher_bvid = all_sort.publisher_bv(BV, params)

            # publisher_data = all_sort.data_crawler(publisher_bvid)
            # rec_data = all_sort.data_crawler(rec_bvid)
            # keyword_data = all_sort.data_crawler(keyword_bvid)

            # video_publisher = all_sort.video_select(publisher_data, title, tag, intro)
            # video_keyword = all_sort.video_select(rec_data, title, tag, intro)
            # video_rec = all_sort.video_select(keyword_data, title, tag, intro)
            # video = pd.concat([video_publisher, video_keyword, video_rec])
            # video.to_csv('相关视频信息.csv')

        if model == '演示模式':
            list = pd.read_csv('示例视频列表.csv', encoding='utf-8')
            list_new = list[['标题', '关键词-自定义', '标签', '简介']]
            col1, col2 = st.columns([1, 1])
            with col1:
                with st.spinner('Wait about 3 seconds'):
                    with st.expander("本地数据库", True):
                        number = st.text_input('请输入演示视频编号:')
                        if number:
                            pic = list['封面图片'].values[int(number)]
                            title = list['标题'].values[int(number)]
                            tag = list['标签'].values[int(number)]
                            intro = list['简介'].values[int(number)]
                            url = list['urls'].values[int(number)]

                            picture = requests.get(pic, stream=True)
                            with open('相关视频' + number + '封面.png', "wb") as png:
                                for chunk in picture.iter_content(chunk_size=1024):
                                    if chunk:
                                        png.write(chunk)
                            st.write(f'<a href="{url}"><p style="font-size:20px;font-weight:bold;">{title}</p ></a >',
                                     unsafe_allow_html=True)
                            col_ex1, col_ex2 = st.columns([1, 1])
                            with col_ex1:
                                st.image('相关视频' + number + '封面.png', caption='视频封面', width=380)
                            with col_ex2:
                                st.write('<p style="font-size:18px;font-weight:bold;">标签：</p >',
                                         unsafe_allow_html=True)
                                st.write(tag)
                                st.write('<p style="font-size:18px;font-weight:bold;">简介：</p >',
                                         unsafe_allow_html=True)
                                st.write(intro)

            with col2:
                with st.expander('可选择视频列表', True):
                    st.write(list_new)
            if url:
                st.write(f'<p style="font-size:25px;font-weight:bold;">此视频相关视频列表:</p >',
                         unsafe_allow_html=True)
                project_path = os.getcwd()
                file = project_path+'\相关视频\第' + number + '条链接相关视频.csv'
                related_video = pd.read_csv(file, encoding='utf-8')
                video = related_video[['标题', '标签', '简介']]
                with st.expander("同作者路径搜索视频:", True):
                    col1_1, col1_2, col1_3, col1_4 = st.columns([1, 1, 1, 1])
                    with col1_1:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">标题相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[0:3])
                    with col1_2:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">标签相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[3:6])
                    with col1_3:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">简介相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[6:9])
                    with col1_4:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">封面相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[9:12])

                with st.expander("大数据路径搜索视频:", True):
                    col2_1, col2_2, col2_3, col2_4 = st.columns([1, 1, 1, 1])
                    with col2_1:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">标题相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[12:15])
                    with col2_2:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">标签相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[15:18])
                    with col2_3:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">简介相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[18:21])
                    with col2_4:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">封面相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[21:24])

                with st.expander("关键词路径搜索视频:", True):
                    col3_1, col3_2, col3_3, col3_4 = st.columns([1, 1, 1, 1])
                    with col3_1:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">标题相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[24:27])
                    with col3_2:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">标签相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[27:30])
                    with col3_3:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">简介相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[30:33])
                    with col3_4:
                        st.write(f'<p style="font-size:15px;font-weight:bold;">封面相似</p >',
                                 unsafe_allow_html=True)
                        st.write(video.iloc[33:36])




if __name__ == '__main__':  # 不用命令端输入“streamlit run app.py”而直接运行
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

