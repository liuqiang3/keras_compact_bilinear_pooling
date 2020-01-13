import numpy as np
import tensorflow as tf
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, Input, Reshape, Lambda,Concatenate

from keras import backend as K
from keras.engine.topology import Layer

class Cbp_layer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Cbp_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重




        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Compact bilinear pooling` layer should be called '
                             'on a list of 2 inputs')
        input1_shape,input2_shape = input_shape


        seed_h_1 = 1
        seed_s_1 = 3
        seed_h_2 = 5
        seed_s_2 = 7
        # Generate sparse_sketch_matrix1 using rand_h_1 and rand_s_1
        np.random.seed(seed_h_1)
        self.input1_dim = input1_shape[-1]
        # self.input1_dim = input1_shape.as_list()[-1]

        # def rand_s1_init(shape, dtype=None,):
        #     s = K.random_normal(shape, dtype=dtype,seed=seed_s_1)
        #     s = tf.cast(tf.floor(s) * 2 - 1, 'int32')  # 1 or -1
        #     return s

        # def rand_s2_init(shape, dtype=None):
        #     s = K.random_normal(shape, dtype=dtype,seed=seed_s_2)
        #     s = tf.cast(tf.floor(s) * 2 - 1, 'int32')  # 1 or -1
        #     return s

        def rand_s1_init(shape, name=None):
            np.random.seed(seed_s_1)
            value = 2*np.random.randint(2, size=shape) - 1
            return K.variable(value, name=name, dtype='float32')


        def rand_s2_init(shape, name=None):
            np.random.seed(seed_s_2)
            value = 2*np.random.randint(2, size=shape) - 1
            return K.variable(value, name=name, dtype='float32')



        self.rand_h_1 = np.random.randint(self.output_dim, size=self.input1_dim,dtype=np.int64)
        np.random.seed(seed_s_1)

        self.rand_s_1 = self.add_weight(name='rand_s_1',
                                      shape=(self.input1_dim,),
                                      initializer=rand_s1_init,
                                      dtype='float32',
                                      trainable=True)

        # Generate sparse_sketch_matrix2 using rand_h_2 and rand_s_2
        self.input2_dim = input2_shape[-1]
        np.random.seed(seed_h_2)
        self.rand_h_2 = np.random.randint(self.output_dim, size=self.input2_dim,dtype=np.int64)
        np.random.seed(seed_s_2)



        self.rand_s_2 = self.add_weight(name='rand_s_2',
                                      shape=(self.input2_dim,),
                                      initializer=rand_s2_init,
                                      dtype='float32',
                                      trainable=True)

        super(Cbp_layer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, tensors_list):
        bottom1, bottom2 = tensors_list
        # Step 1: Flatten the input tensors and count sketch

        bottom1_flat = tf.reshape(bottom1, [-1, self.input1_dim])
        bottom2_flat = tf.reshape(bottom2, [-1, self.input2_dim])

        #   sketch1 = bottom1 * sparse_sketch_matrix
        #   sketch2 = bottom2 * sparse_sketch_matrix
        # But tensorflow only supports left multiplying a sparse matrix, so:
        #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
        #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
        sparse_sketch_matrix1 = self._generate_sketch_matrix(self.rand_h_1, self.rand_s_1, self.output_dim)
        sparse_sketch_matrix2 = self._generate_sketch_matrix(self.rand_h_2, self.rand_s_2, self.output_dim)

        sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
                                                             bottom1_flat, adjoint_a=True, adjoint_b=True))
        sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
                                                             bottom2_flat, adjoint_a=True, adjoint_b=True))

        # Step 2: FFT
        fft1 = tf.fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
        fft2 = tf.fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

        # Step 3: Elementwise product
        fft_product = tf.multiply(fft1, fft2)

        # Step 4: Inverse FFT and reshape back
        # Compute output shape dynamically: [batch_size, height, width, output_dim]
        cbp_flat = tf.real(tf.ifft(fft_product))

        output_shape = tf.add(tf.multiply(tf.shape(bottom1), [1, 1, 1, 0]),
                              [0, 0, 0, self.output_dim])
        cbp = tf.reshape(cbp_flat, output_shape)
        # set static shape for the output
        cbp.set_shape(bottom1.get_shape().as_list()[:-1] + [self.output_dim])
        # print(cbp.get_shape)
        # Step 5: Sum pool over spatial dimensions, if specified
        cbp = tf.reduce_sum(cbp, reduction_indices=[1, 2])
        # print(cbp.get_shape())
        return cbp

    def compute_output_shape(self, input_shape):
        print(input_shape[0], self.output_dim)
        return (input_shape[0][0], self.output_dim)

    def _generate_sketch_matrix(self,rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch

        assert (rand_h.ndim == 1 and K.ndim(rand_s) == 1 and len(rand_h) == rand_s.get_shape()[0])
        assert (np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        sparse_sketch_matrix = tf.sparse_reorder(
            tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
        return sparse_sketch_matrix


