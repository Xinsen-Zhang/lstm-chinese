import codecs
import re
import numpy as np
from config import BATCH_SIZE as batch_size
from config import TIME_STEPS as time_steps
from sklearn.preprocessing import OneHotEncoder

# =============== 读取文本内容 ===============
f = codecs.open('./data/new.txt', 'r', encoding='utf-8')
data = f.readlines()
# data = ''.join(data)

#=============== 简单的预处理 ===============
# 替换括号里的内容
pattern = re.compile(r'\(.*?\)')
data = [pattern.sub('', line) for line in data]

# 删除\n, \r,' '
data = [word.replace('.', '。') for word in data]
data = [word.replace('\r', '') for word in data]
# data = [word.replace(' ', '') for word in data]

# 删除章节名称
pattern = re.compile(r'.*?第.*?章.*')
def isNotChapterName(text):
    if pattern.search(text):
        return False
    else:
        return True
data = [word for word in data if isNotChapterName(word)]

# 省略号 => .
data = [line.replace('……', '。') for line in data if len(line) > 1]

# ==============判断char是否是乱码===================
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True       
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('，','。','：','？','“','”','！','；','、','《','》','——'):
            return True
    return False

# 将每行的list合成一个长字符串
data = ''.join(data)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)



# ==============生成字典===============
vocab = set(data)
id2char = {i:c for i,c in enumerate(vocab)}
char2id = {c:i for i,c in enumerate(vocab)}
# 总数
VOCAB_NUM = len(id2char)


# =====转换数据为数字格式======
numdata = [char2id[char] for char in data]
numdata = np.array(numdata)
batch_num = len(numdata) // (batch_size * time_steps)
# print(numdata.shape)

# print('数字数据信息：\n', numdata[:100])
# print('\n文本数据信息：\n', ''.join([id2char[i] for i in numdata[:100]]))
# print(len(numdata))

# ============= 定义 dataloader =============
def yield_data( batch_size, time_step, data= numdata):
    encoder = OneHotEncoder(categories= [range(VOCAB_NUM)])
    start = [i * time_step for i in range(batch_size)]
    end = [(i+1) * time_step for i in range(batch_size)]
    data_num = batch_size * time_step
    batch_num = len(data) // data_num
    arr = data[:data_num * batch_num]
    arr_y = np.roll(arr,-1)
    arr = arr.reshape((batch_size, -1))
    arr_y = arr_y.reshape((batch_size, -1))
    for i in range(0,arr.shape[1], time_step):
        x = arr[:,i:i + time_step].reshape((batch_size, -1))
        x = encoder.fit_transform(x.reshape(-1,1)).toarray().reshape(x.shape[0], x.shape[1], -1)
        y = arr_y[:,i:i + time_step].reshape((batch_size, -1))
        yield (np.array(x, dtype= np.float),y)

if __name__ == '__main__':
    data = yield_data(128,100)
    for x,y in data:
        break
    print(x.shape)
    print(y.shape)