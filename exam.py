# 打开pickle
# data_film = open('./data/toy_dataset.pickle', 'rb')
# data = pickle.load(data_film)
# print(data)


# json文件，将数据导入excel
# data = open('./data/Electronics_5.json')
# datas = []
# for line in data.readlines():
#     dic = json.loads(line)  # 将json格式数据转换为字典
#     datas.append(dic)
# normal = pandas.json_normalize(datas)  # 将json串解析为DataFrame     # data = pd.DataFrame(datas)

# normal['reviewTime'] = pd.to_datetime(normal['reviewTime'])  # 转为时间格式
# print(normal)
# normal.tail(689188).to_excel('./data/data2.xlsx', engine='xlsxwriter')

# df = pd.read_json("./data/new2.json", encoding="utf-8", orient='records')


# csv文件
# dir_data = './data/ratings_Electronics.csv'
# df = pd.read_csv(dir_data, header=None, names=['user', 'item', 'rating', 'time'])
# print(df.tail())


# ###########################################################################################, 'r', encoding='utf-8'
# 解析json，写入csv,读csv

# data = open('./data/Electronics_5.json')
# datas = []
# for line in data.readlines():
#     dic = json.loads(line)  # 将json格式数据转换为字典
#     datas.append(dic)
# normal = pandas.json_normalize(datas)  # 将json串解析为DataFrame
# normal['reviewTime'] = pd.to_datetime(normal['reviewTime'])  # 转为时间格式
# data_2013 = normal[(normal['reviewTime'] >= pd.to_datetime('20130101')) & (normal['reviewTime'] <= pd.to_datetime('20130131'))]
# data_2013_0 = data_2013.reset_index(drop=True)
# data_2013_0 = data_2013_0.drop('reviewText', axis=1)
# data_2013_0 = data_2013_0.drop('reviewerName', axis=1)
# data_2013_0 = data_2013_0.drop('helpful', axis=1)
# data_2013_0 = data_2013_0.drop('summary', axis=1)
# data_2013_0 = data_2013_0.drop('unixReviewTime', axis=1)
# data_2013_0.to_csv('./data/2013drop.csv', index=False, header=False)
#
# df = pd.read_csv('./data/2013-12.csv', header=None, names=['reviewerID', 'item', 'rating', 'time'])
# df['time'] = pd.to_datetime(df['time'])
#
#
# data_2013_1 = df[(df['time'] >= pd.to_datetime('20131201')) & (df['time'] <= pd.to_datetime('20131231'))]
# data_2013_1 = data_2013_1.reset_index(drop=True)
# data_2013_1.to_csv('./data/2013-12.csv', index=False, header=False)
# ################################################################################################################

# df = df.sort_values(by="user", ascending=False)  # 聚集
# da = df.groupby('user').filter(lambda x: (len(x) > 4))  # 过滤
# user = data_2016[data_2016['userId'] == 162516]

# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
# ##################################################################################################################

# 将txt中的字典读入
# data_film = open('./data/test.txt', 'r')
# history_u_list = eval(data_film.read())
# print(history_u_list)
# ###################################################################################################################

# 合并数据集
# dir_data = './data/epinions/epinions.train.rating'
# dir_data_test = './data/epinions/epinions.test.rating'
# df = pd.read_table(dir_data, header=None, names=['user', 'item', 'rating', 'time'])
# dt = pd.read_table(dir_data_test, header=None, names=['user', 'item', 'rating', 'time'])
# data = pd.concat([df, dt])
# data = data.reset_index(drop=True)
# data = data.sort_values(by="user", ascending=True)
# data = data.reset_index(drop=True)

# #################################################################################################################

# 分组后，将time列按大小排序
# da = df.groupby('user').apply(lambda x: x.sort_values('time', ascending=False))
# df = df.groupby('user', group_keys=False).apply(lambda x: x.sort_values('time', ascending=False))
#
# da = dict(df['user'].value_counts()).__len__()

# 分组后存入字典，写入文件
# dir_data = './data/epinions/sort_fil.csv'
# df = pd.read_csv(dir_data, header=None, names=['user', 'item', 'rating', 'time'])
# GroupBy = df.groupby('user')
# # dic = {}
# # for num, group in GroupBy:
# #     dic.update({num: list(group['item'])})
# #     # print(num)
# #     # print(list(group['item']))
# # print(dic)
# # with open('./data/1.pkl', 'wb') as file:
# #     pickle.dump(dic, file)


# df = df[(df['time'] == 12)]
# list_u = list(df['user'])
# list_v = list(df['item'])
# list_r = list(df['rating'])
# with open('./data/test_u.pkl', 'wb') as file:
#     pickle.dump(list_u, file)

# ##############################################################################################
# 分组后重新给id赋值，使其连续
# dir_data = './data/epinions/sort_fil.csv'
# df = pd.read_csv(dir_data, header=None, names=['user', 'item', 'rating', 'time'])
# test = pd.DataFrame(columns=['user', 'item', 'rating', 'time'])
# # GroupBy = df.groupby('item')
# # for i, group in enumerate(GroupBy, 0):
# #     x = group[1].copy()
# #     x['item'].replace(group[0], i, inplace=True)
# #     test = test.append(x)

# ###############################################################################################
# 读txt文件
# dir_data = './data/CiaoDVD/movie-ratings.txt'
# df = pd.read_table(dir_data, header=None, names=['user', 'item', 'id', 'reid', 'rating', 'time'], sep=',')


# train(mtsrec, device, train_loader, optimizer, epoch, best_rmse, best_mae, history_u_lists_1, history_ur_lists_1,
#               history_u_lists_2, history_ur_lists_2, history_u_lists_3, history_ur_lists_3, history_u_lists_4, history_ur_lists_4,
#               history_u_lists_5, history_ur_lists_5, history_u_lists_6, history_ur_lists_6, history_u_lists_7, history_ur_lists_7,
#               history_u_lists_8, history_ur_lists_8, history_u_lists_9, history_ur_lists_9, history_u_lists_10, history_ur_lists_10,
#               history_u_lists_11, history_ur_lists_11, history_u_lists_12, history_ur_lists_12, history_v_lists, history_vr_lists)
#         expected_rmse, mae = test(mtsrec, device, test_loader, history_u_lists_1, history_ur_lists_1,
#               history_u_lists_2, history_ur_lists_2, history_u_lists_3, history_ur_lists_3, history_u_lists_4, history_ur_lists_4,
#               history_u_lists_5, history_ur_lists_5, history_u_lists_6, history_ur_lists_6, history_u_lists_7, history_ur_lists_7,
#               history_u_lists_8, history_ur_lists_8, history_u_lists_9, history_ur_lists_9, history_u_lists_10, history_ur_lists_10,
#               history_u_lists_11, history_ur_lists_11, history_u_lists_12, history_ur_lists_12, history_v_lists, history_vr_lists)


# u_review = TextCNN(opts.word_dim, opts.vocab_size, opts.num_filters, opts.filter_sizes, opts.seq_len)

# #########################################################################################################
# data['unixReviewTime'] = pd.to_datetime(data['unixReviewTime'], unit='s')  # 转为时间格式

# data['overall'] = pd.to_numeric(data['overall'])
# data['overall'] = data['overall'].apply(lambda x: x - 1.0)

# data.to_csv('../dataset/Electronics.csv', index=False, header=False)

# df = df.sort_values(by="time", ascending=False)

# data_2019 = df[(df['time'] >= '2019-01-01') & (df['time'] <= '2019-12-31')]

# #####################################################################################################
# df_time = data['time']
# df = data.drop('time', axis=1)
# df.insert(1, 'time', df_time)
# df.to_csv('../dataset/filter_0.csv', index=False, header=False)



# df = df.sort_values(by="user", ascending=False)  # 聚集
# da = df.groupby('user').filter(lambda x: (len(x) > 10))  # 过滤
# data = da.groupby('user').apply(lambda x: x.sort_values('time', ascending=False))
# data.to_csv('../dataset/filter.csv', index=False, header=False)

# ########################################################################################################
# data = pd.read_csv('../dataset/Electronics_2014_filter.csv', header=None, names=['user', 'item', 'text', 'rating', 'time'])
#
# res = pd.DataFrame(columns=('user', 'item', 'text', 'rating', 'time'))  # 初始化一个空的dataframe
# GroupBy = data.groupby('user')
# user = 0
#
# for num in GroupBy:
#     user = user + 1
#     time_1 = 0
#     time_2 = 0
#     time_3 = 0
#     time_4 = 0
#     time_5 = 0
#     time_6 = 0
#     for row in num[1].itertuples(index=True, name='Pandas'):
#         if (getattr(row, 'time') >= '2014-01-01') & (getattr(row, 'time') <= '2014-02-28'):
#             if time_1 == 1:
#                 continue
#             else:
#                 time_1 = 1
#                 continue
#         if (getattr(row, 'time') >= '2014-03-01') & (getattr(row, 'time') <= '2014-04-30'):
#             if time_2 == 1:
#                 continue
#             else:
#                 time_2 = 1
#                 continue
#         if (getattr(row, 'time') >= '2014-05-01') & (getattr(row, 'time') <= '2014-06-30'):
#             if time_3 == 1:
#                 continue
#             else:
#                 time_3 = 1
#                 continue
#         if (getattr(row, 'time') >= '2014-07-01') & (getattr(row, 'time') <= '2014-08-31'):
#             if time_4 == 1:
#                 continue
#             else:
#                 time_4 = 1
#                 continue
#         if (getattr(row, 'time') >= '2014-09-01') & (getattr(row, 'time') <= '2014-10-31'):
#             if time_5 == 1:
#                 continue
#             else:
#                 time_5 = 1
#                 continue
#         if (getattr(row, 'time') >= '2014-11-01') & (getattr(row, 'time') <= '2014-12-31'):
#             if time_6 == 1:
#                 continue
#             else:
#                 time_6 = 1
#                 continue
#     if time_1 & time_2 & time_3 & time_4 & time_5 & time_6:
#         res = pd.concat([res, num[1]], axis=0)  # 上下合并两个dataframe
# ######################################################################################################
# data_1 = yelp_2018[(yelp_2018['time'] >= '2018-01-01') & (yelp_2018['time'] <= '2018-02-28')]
# data_2 = yelp_2019[(yelp_2019['time'] >= '2019-03-01') & (yelp_2019['time'] <= '2019-04-30')]
# data_3 = yelp_2019[(yelp_2019['time'] >= '2019-05-01') & (yelp_2019['time'] <= '2019-06-30')]
# data_4 = yelp_2019[(yelp_2019['time'] >= '2019-07-01') & (yelp_2019['time'] <= '2019-08-31')]
# data_5 = yelp_2019[(yelp_2019['time'] >= '2019-09-01') & (yelp_2019['time'] <= '2019-10-31')]
# data_6 = yelp_2019[(yelp_2019['time'] >= '2019-11-01') & (yelp_2019['time'] <= '2019-12-31')]

# data_1 = yelp_2018[(yelp_2018['time'] >= '2018-01-01') & (yelp_2018['time'] <= '2018-01-31')]
# data_2 = yelp_2018[(yelp_2018['time'] >= '2018-02-01') & (yelp_2018['time'] <= '2018-02-28')]
# data_3 = yelp_2018[(yelp_2018['time'] >= '2018-03-01') & (yelp_2018['time'] <= '2018-03-31')]
# data_4 = yelp_2018[(yelp_2018['time'] >= '2018-04-01') & (yelp_2018['time'] <= '2018-04-30')]
# data_5 = yelp_2018[(yelp_2018['time'] >= '2018-05-01') & (yelp_2018['time'] <= '2018-05-31')]
# data_6 = yelp_2018[(yelp_2018['time'] >= '2018-06-01') & (yelp_2018['time'] <= '2018-06-30')]
# data_7 = yelp_2018[(yelp_2018['time'] >= '2018-07-01') & (yelp_2018['time'] <= '2018-07-31')]
# data_8 = yelp_2018[(yelp_2018['time'] >= '2018-08-01') & (yelp_2018['time'] <= '2018-08-31')]
# data_9 = yelp_2018[(yelp_2018['time'] >= '2018-09-01') & (yelp_2018['time'] <= '2018-09-30')]
# data_10 = yelp_2018[(yelp_2018['time'] >= '2018-10-01') & (yelp_2018['time'] <= '2018-10-31')]
# data_11 = yelp_2018[(yelp_2018['time'] >= '2018-11-01') & (yelp_2018['time'] <= '2018-11-30')]
# data_12 = yelp_2018[(yelp_2018['time'] >= '2018-12-01') & (yelp_2018['time'] <= '2018-12-31')]

# data_1.to_csv('../dataset/yelp_2019_1.csv', index=False, header=False)

# item_meta[i[1]].append((i[0], i[2], float(i[3])))
# ######################################################################################################
# user_meta = {}  # data中每一个用户对应的项目们,评分们，评论们
# # item_meta = {}  # data中每一个项目对应的用户们,评分们，评论们
#
# for i in data_1.values:
#     if i[0] in user_meta:
#         user_meta[i[0]].append((i[1], i[2], float(i[3])))
#     else:
#         user_meta[i[0]] = [(i[1], i[2], float(i[3]))]  # data中第i行对应的用户id,项目id，用户对项目的评论
#     # if i[1] in item_meta:
#     #     item_meta[i[1]].append((i[0], float(i[2]), i[3]))
#     # else:
#     #     item_meta[i[1]] = [(i[0], float(i[2]), i[3])]  # data中第i行对应的项目id，用户id, 用户对项目的评论
#
#
# user_rid = {}  # 用户id，及用户交互的所有项目id
# user_rid_ra = {}  # 用户id，及用户交互的所有项目rating
# user_reviews = {}   # 用户id，及用户给出的所有评论
#
# # item_rid = {}  # 项目id，及项目交互的所有用户id
# # item_rid_ra = {}  # 项目id，及项目交互的所有项目rating
# # item_reviews = {}  # 项目id，及对应用户们给出的所有评论
#
# for u in user_meta:
#     user_rid[u] = [i[0] for i in user_meta[u]]
#     user_rid_ra[u] = [int(i[2]) for i in user_meta[u]]
#     user_reviews[u] = [(i[1]) for i in user_meta[u]]
# # for i in item_meta:
# #     item_rid[i] = [x[0] for x in item_meta[i]]
# #     item_rid_ra[i] = [int(x[1]) for x in item_meta[i]]
# #     item_reviews[i] = [(x[2]) for x in item_meta[i]]
#
# print("done")
# pickle.dump(user_reviews, open('../dataset/Elec2013_user_review_6.pkl', 'wb'))  # 用户id，及用户给出的所有评论
# # pickle.dump(item_reviews, open('../dataset/yelp_2018_item_review.pkl', 'wb'))  # 项目id，及对应用户们给出的所有评论
# pickle.dump(user_rid, open('../dataset/Elec2013_user_rid_6.pkl', 'wb'))  # 用户id，及用户交互的所有项目id
# # pickle.dump(item_rid, open('../dataset/yelp_2018_item_rid.pkl', 'wb'))  # 项目id，及项目交互的所有用户id
# pickle.dump(user_rid_ra, open('../dataset/Elec2013_user_rid_ra_6.pkl', 'wb'))  # 用户id，及用户交互的所有项目rating
# # pickle.dump(item_rid_ra, open('../dataset/yelp_2018_item_rid_ra.pkl', 'wb'))  # 项目id，及项目交互的所有项目rating