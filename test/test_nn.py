from nn import nn, preprocess
import numpy as np

arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
lr = 0.001
batch_size = 128


def test_single_forward():
    mynn = nn.NeuralNetwork(nn_arch = arch, 
                            lr = lr,
                            seed=0,
                            batch_size=batch_size,
                            epochs=10,
                            loss_function='mse'
                           )
    w = np.array([[1,1]])
    b = np.array([[2]])
    a = np.array([[2,2]])
    
    a,z = mynn._single_forward(w, b, a, 'sigmoid')
    
    assert a[0][0] == 0
    assert z[0][0] == 6

def test_forward():
    X = np.random.randint(0,100, size=(1200,64))
    y = np.random.randint(0,100, (1200,8))

    arch = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
    lr = 0.0001
    seed = 0
    batch_size=128
    epochs = 20
    loss_function = 'mse'

    mynn = nn.NeuralNetwork(nn_arch = arch, lr = lr, seed = seed, batch_size=batch_size, epochs=epochs, loss_function=loss_function)
    y, cache = mynn.forward(X)

    assert y.shape == (1200,8)
    for i in ['a_-1', 'a_0', 'z_0', 'a_1', 'z_1']:
        assert i in cache.keys()
    

def test_single_backprop():
    
    arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 0.001
    batch_size = 128
    mynn = nn.NeuralNetwork(nn_arch = arch, 
                                lr = lr,
                                seed=0,
                                batch_size=batch_size,
                                epochs=10,
                                loss_function='mse'
                               )
    wcurr = np.array([[1,4]])
    bcurr = np.array([[3]])
    zcurr = np.array([[1]])
    da_curr = np.array([1])
    a_prev = np.array([[1,3]])
    activation_curr='relu'

    daprev, dwcurr, dbcurr = mynn._single_backprop(wcurr, bcurr, zcurr, a_prev,da_curr,activation_curr)

    np.testing.assert_allclose(daprev, np.array([[1,4]]), rtol=1e-2)
    np.testing.assert_allclose(dwcurr, np.array([[1,3]]), rtol=1e-2)
    np.testing.assert_allclose(dbcurr, np.array([1]), rtol=1e-2)

    activation_curr='sigmoid'
    daprev, dwcurr, dbcurr = mynn._single_backprop(wcurr, bcurr, zcurr, a_prev,da_curr,activation_curr)
    np.testing.assert_allclose(daprev, np.array([[0,0]]), rtol=1e-2)
    np.testing.assert_allclose(dwcurr, np.array([[0,0]]), rtol=1e-2)
    np.testing.assert_allclose(dbcurr, np.array([0]), rtol=1e-2)


def test_predict():
    X = np.random.randint(0,100, size=(1200,64))
    y = np.random.randint(0,100, (1200,8))

    arch = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
    lr = 0.0001
    seed = 0
    batch_size=128
    epochs = 20
    loss_function = 'mse'

    mynn = nn.NeuralNetwork(nn_arch = arch, lr = lr, seed = seed, batch_size=batch_size, epochs=epochs, loss_function=loss_function)
    y = mynn.predict(X)
    
    assert y.shape == (1200,8)


def test_binary_cross_entropy():
    arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 0.001
    batch_size = 128
    mynn = nn.NeuralNetwork(nn_arch = arch, 
                                lr = lr,
                                seed=0,
                                batch_size=batch_size,
                                epochs=10,
                                loss_function='mse'
                               )
    bce = mynn._binary_cross_entropy(np.array([[0,0]]), np.array([[0,1]])) 
    assert bce == 4.605120188487924


def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    lr = 0.001
    batch_size = 128
    mynn = nn.NeuralNetwork(nn_arch = arch, 
                            lr = lr,
                            seed=0,
                            batch_size=batch_size,
                            epochs=10,
                            loss_function='mse'
                           )
    assert mynn._mean_squared_error(np.array([2]), np.array([1])) == 1


def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    seq = ['ATCG']
    seq_oh = preprocess.one_hot_encode_seqs(seq)
    test_list = list('1000010000100001')
    np.testing.assert_array_equal(seq_oh[0], test_list)