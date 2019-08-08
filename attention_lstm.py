from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

from attention_utils import get_activations, get_data_recurrent

# INPUT_DIM 是词向量的维度
INPUT_DIM = 2

# TIME_STEPS 理解有点类似一句话有多少个词
TIME_STEPS = 20

# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])

    # Permute用于把向量维度转换，这里面是把(20,32)的样本向量，变成(32,20)维度。
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.

    # 这里是在训练注意力，但是咋只有一个20维呢？
    a = Dense(TIME_STEPS, activation='softmax')(a)

    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1), name='attention_vec')(a)

    # inputs和a_probs维度是一样的，对应位置的元素相乘，就是Attention的原理
    # merge合并，支持多种方式，如 'sum', 'mul', 'concat', 'ave', 'cos', 'dot', 'max'
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))

    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)

    # 构建注意力系数与输入的结果
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    # 输出二分类概率
    output = Dense(1, activation='sigmoid')(attention_mul)

    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':

    N = 300000
    # N = 300 -> too few = no training

    # 创建300000个样本，第10位的样本是两个相同的数字，跟label刚好是匹配的。
    # TODO 这样训练的目的，应该就是想让模型聚焦到第10位，根据第10位的内容进行预测y
    inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)

    # 是在LSTM之前用Attention还是之后？TODO 暂时不明白两者的区别
    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm()

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(300):
        testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=2).squeeze()
        print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()
