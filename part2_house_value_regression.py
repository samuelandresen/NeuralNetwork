import torch
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.base import BaseEstimator
import torch.nn as nn


class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.model = None
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.logits = []
        self.learning_rate = 0.0011
        self.batch_size = 30
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        df = pd.DataFrame(x)
        self.normalised_constant = 0
        self.mapping_constant = 0
        df = df.fillna(0)
        lb = preprocessing.LabelBinarizer()
        a = lb.fit_transform(df["ocean_proximity"]).T
        ctr = 0
        for i in (lb.classes_):
            df[i] = a[ctr]
            ctr += 1

        df.drop("ocean_proximity", axis = 'columns', inplace = True)

        def normalization(dataframe):
            for column in dataframe.columns:
                
                dataframe[column] = (dataframe[column] - dataframe[column].min()) / (dataframe[column].max() - dataframe[column].min())    
                
            return dataframe
        
        
        df = normalization(df)
        # df = lb.fit(df)
        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        df = np.array(df)
        y = np.array(y)
        return df, (y if isinstance(y, np.ndarray) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        seq_modules = nn.Sequential(
                nn.Linear(X.shape[1],1),
                nn.ReLU(),
                # nn.Linear(6,1)
                # nn.ReLU(),
                # nn.Linear(3,1)
            )

        def forward(x):
            logits = seq_modules(x)
            return logits

        optimizer = torch.optim.SGD(seq_modules.parameters(), lr = self.learning_rate)
        
        def training_loop(X, Y, model, optimizer):
            batch_num = self.batch_size
            X_batches = np.array_split(X, batch_num)
            Y_batches = np.array_split(Y, batch_num)

            for i in range(batch_num):
                X_batch = X_batches[i]
                Y_batch = Y_batches[i]

                X_tensor = torch.tensor(X_batch, requires_grad=True).to(torch.float32)
                Y_tensor = torch.tensor(Y_batch, requires_grad=True).to(torch.float32)
                self.logits = self.model(X_tensor)

                # loss
                loss = 0.5 * ((self.logits - Y_tensor) ** 2).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for i in range(self.nb_epoch):

            training_loop(X, Y, forward, optimizer)
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training = False) # Do not forget
        # regressor = Regressor(x, nb_epoch = 10)
        # self = regressor.fit(x,X)
        # save_regressor(regressor)
        X = torch.tensor(X).to(torch.float32)
        logits = self.model(X)
        logits_array = logits.detach().numpy()
        self.logits = logits_array
        return logits_array

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        prediction = Regressor.predict(self, x)
        y_tensor = torch.tensor(Y)
        prediction_tensor = torch.tensor(prediction)
        score = torch.sqrt(torch.mean(((prediction_tensor - y_tensor)**2)))
        score_array = score.detach().numpy()
        score_float = score_array.astype(np.float32)
        return score_float# Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(input_data, output_data, parameters): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    len_fold = len(input_data.index) // 10
    training_fold = torch.randperm(len(input_data.index))[len_fold * 9:]
    development_fold = torch.randperm(len(input_data.index))[len_fold * 8:len_fold * 9]

    input_train = input_data.iloc[training_fold]

    input_development = input_data.iloc[development_fold]
    output_development = output_data.iloc[development_fold]

    model = model_selection.GridSearchCV(
        Regressor(x = input_train), 
        param_grid = parameters, 
        scoring = 'neg_root_mean_squared_error',
        verbose = 2, 
        return_train_score=True)

    model.fit(input_development, output_development)

    return model.best_estimator_.learning_rate, model.best_estimator_.batch_size, model.best_estimator_.nb_epoch 

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 500)
    regressor.data = data
    regressor.fit(x_train, y_train)
    save_regressor(regressor)
    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()


