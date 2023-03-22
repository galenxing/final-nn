# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        z = np.dot(A_prev, W_curr.T) + b_curr.T
        
        if activation == 'sigmoid':
            a = self._sigmoid(z)
        elif activation == 'relu':
            a = self._relu(z)
        
        return (a, z)

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        for i, arch in enumerate(self.arch):
            if i == 0:
                prev_a = X
                cache['a_-1']=prev_a
            else:
                prev_a = curr_a
    
            curr_w = self._param_dict[f'W{i+1}']
            curr_b = self._param_dict[f'b{i+1}']
            activation = arch['activation']

            curr_a, curr_z = self._single_forward(curr_w, curr_b, prev_a, activation=activation)
            cache[f'a_{i}'] = curr_a
            cache[f'z_{i}'] = curr_z
            
        return curr_a, cache
    
    
    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        
        if activation_curr == 'sigmoid':
            dz_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'relu':
            dz_curr = self._relu_backprop(dA_curr, Z_curr)
        else:
            raise ValueError('Activation function needs to be "sigmoid" or "relu"')
    
#         dz_curr = self._get_loss(dA_curr, Z_curr, is_backprop=True)
    
        import pdb
#         pdb.set_trace()
        da_prev = np.dot(dz_curr, W_curr)
        dw_curr = np.dot(dz_curr.T, A_prev)
        db_curr = np.sum(dz_curr, axis=0)
        
        return da_prev, dw_curr, db_curr
        
    
    
    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
            this is mine
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grads = {}
        import pdb
        da_curr = self._get_loss(y,y_hat, is_backprop=True)        
        for i, arch in enumerate(self.arch[::-1]):


            i = len(self.arch)-i
            w_curr = self._param_dict[f'W{i}']
            b_curr = self._param_dict[f'b{i}']
            z_curr = cache[f'z_{i-1}']
            
#             pdb.set_trace()
            a_prev = cache[f'a_{i-2}']
            
            activation = arch['activation']
            
            da_prev, dw_curr, db_curr = self._single_backprop(
                W_curr = w_curr, 
                b_curr = b_curr, 
                Z_curr = z_curr,
                A_prev = a_prev, 
                dA_curr = da_curr, 
                activation_curr = activation)
            
            da_curr = da_prev
#             pdb.set_trace()            
            grads[f'db_{i-1}'] = db_curr
            grads[f'dw_{i-1}'] = dw_curr
        
        return grads

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for i,_ in enumerate(self.arch):
            self._param_dict[f'W{i+1}'] -= self._lr * grad_dict[f'dw_{i}']
            self._param_dict[f'b{i+1}'] -= self._lr * grad_dict[f'db_{i}'].reshape((-1,1))

        
    def _get_loss(self,y, y_hat, is_backprop=False):
        if is_backprop:
            if self._loss_func == 'mse':
                val = self._mean_squared_error_backprop(y, y_hat)
            elif self._loss_func == 'bce':
                val = self._binary_cross_entropy_backprop(y, y_hat)
            else:
                raise ValueError('loss function needs to be "mse" or "bce"')
        elif is_backprop==False:
            if self._loss_func == 'mse':
                val = self._mean_squared_error(y, y_hat)
            elif self._loss_func == 'bce':
                val = self._binary_cross_entropy(y, y_hat)
            else:
                raise ValueError('loss function needs to be "mse" or "bce"')
        return val
        
        
    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        
        for i in range(self._epochs):
            tmp_loss = []
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            for i in range(0, y_train.shape[0], int(self._batch_size)):

                xbatch, ybatch = X_train[i:i+128], y_train[i:i+128]
                yhat, cache =self.forward(xbatch)
                loss = self._get_loss(ybatch, yhat)
                tmp_loss.append(loss)
                grads  =self.backprop(y_hat=yhat, y=ybatch, cache=cache)
                self._update_params(grads)
            per_epoch_loss_train.append(np.mean(tmp_loss))
            yhat = self.predict(X_val)
            val_loss = self._get_loss(y=y_val, y_hat=yhat)
            per_epoch_loss_val.append(val_loss)
            
        return per_epoch_loss_train, per_epoch_loss_val
            

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y, _ = self.forward(X)
        return y

#     def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
#         """
#         Sigmoid activation function.

#         Args:
#             Z: ArrayLike
#                 Output of layer linear transform.

#         Returns:
#             nl_transform: ArrayLike
#                 Activation function output.
#         """
#         return 1/(1+np.exp(-Z))

    def _positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def _negative_sigmoid(self, x):
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)
        return exp / (exp + 1)


    def _sigmoid(self,Z):
        # stole this from somewhere on stackoverflow cuz of overflow errors
        x =Z
        positive = x >= 0
        # Boolean array inversion is faster than another comparison
        negative = ~positive

        # empty contains juke hence will be faster to allocate than zeros
        result = np.empty_like(x)
        result[positive] = self._positive_sigmoid(x[positive])
        result[negative] = self._negative_sigmoid(x[negative])

        return result

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        ds = self._sigmoid(Z) * (1 - self._sigmoid(Z))
        return dA * ds

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return ((Z > 0)+1e-7) * dA

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        loss = ((1-y)*np.log(1-y_hat + 1e-4)) + (y * np.log(y_hat + 1e-4))
        return -np.mean(loss)

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        y_hat = np.clip(y_hat, 1e-4, 1-1e-4)
        da = ((1-y)/(1-y_hat))- (y/y_hat)
        return da/len(y)

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        mse = (y-y_hat)**2
        return np.mean(mse)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        da = (2* (y_hat-y))
        return da/len(y)
    
    