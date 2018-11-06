import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from load_data import get_example_data



# linear
def linear_transform(X,Y):
    return X,Y

def linear_transform_inv(Y_hat):
    return Y_hat

# exponential
def exponential_transform(X,Y):
    return X, np.log(Y)

def exponential_transform_inv(Y_hat):
    return np.exp(Y_hat)

# logarithmic
def logarithmic_transform(X,Y):
    return X, np.exp(Y)

def logarithmic_transform_inv(Y_hat):
    return np.log(Y_hat)

# logarithmic2, uses linear inv
def logarithmic_transform2(X,Y):
    return np.log(X), Y

# quadratic
def quadratic_transform(X,Y):
    return X, np.sqrt(Y)

def quadratic_transform_inv(Y_hat):
    return Y_hat**2

# power
def power_transform(X,Y):
    return np.log(X), np.log(Y)

def power_transform_inv(Y_hat):
    return np.exp(Y_hat)

# loglogX logY, uses power_transform_inv
def loglogX_logY_transform(X,Y):
    return np.log(np.log(X)), np.log(Y)

def logInvSqrtY_transform(X,Y):
    #import pdb; pdb.set_trace()
    return X, np.log(1/(Y**0.5))

def logInvSqrtY_transform_inv(Y_hat):
    return [0] * len(Y_hat)

# uses linear inv
def logInvSqrtX_transform(X,Y):
    return np.log(1/(X**0.5)), Y

def sqrtInvLogY_transform(X,Y):
    import pdb; pdb.set_trace()
    # taking sqrt of negative number doesn't work
    return X, np.sqrt(np.log(Y)**(-0.5))

# uses linear inv
def sqrtInvLogX_transform(X,Y):
    return np.sqrt(np.log(X)**(-0.5)), Y


def get_transformations():
    return {
        #"linear": [linear_transform, linear_transform],
        "exponential": [exponential_transform, exponential_transform_inv],
        "logarithmic": [logarithmic_transform, logarithmic_transform_inv],
        "quadratic": [quadratic_transform, quadratic_transform_inv],
        "power": [power_transform, power_transform_inv],
        "log(log(X)), log(Y)": [loglogX_logY_transform, power_transform_inv],
        "log(1/Y^0.5)": [logInvSqrtY_transform,logInvSqrtY_transform_inv], 
        "log(1/X^0.5)": [logInvSqrtX_transform, linear_transform_inv],
        #"(log(Y))^{-0.5}": sqrtInvLogY_transform,
        "(log(X))^{-0.5}": [sqrtInvLogX_transform, linear_transform_inv]
    }

def main():
    train, test = get_example_data()
    X = train[:,0].reshape(-1,1)
    Y = train[:,1]


    matplotlib.rcParams.update({'font.size':5})
    fig = plt.figure()



    counter = 0
    for t_name, transform in get_transformations().items():
        counter += 1

        # Create linear regression object
        regr = linear_model.LinearRegression()
        print("about to try " + t_name)
        cur_X, cur_Y = transform[0](X, Y)
    
        # Train the model using the training sets
        regr.fit(cur_X, cur_Y)
        
        
        #import pdb; pdb.set_trace()
        # Make predictions using the testing set

        X_test = test[:,0].reshape(-1,1)
        cur_X_test, _ = transform[0](X_test, Y)
        
        cur_Y_pred = transform[1](regr.predict(cur_X_test))

        cum_abs_error = 0

        for i in range(len(X_test)):
            print("{}:{},".format(int(X_test[i][0]), cur_Y_pred[i]))
        #for item in zip(X_test, cur_Y_pred, test[:,1]):
        #    print(item, item[1]-item[2])
        #    cum_abs_error += abs(item[1]-item[2])
            
        #print("cumulative absolute error: {:.7f}".format(cum_abs_error))
        

        # R^2
        print(regr.score(cur_X, cur_Y))
        
        
        Y_train_preds = regr.predict(cur_X)    
        

        cur_ax = fig.add_subplot(3,3,counter)
        # Plot outputs
        cur_ax.scatter(cur_X, cur_Y,  color='black')
        cur_ax.plot(cur_X, Y_train_preds, color='blue', linewidth=3)
        
        cur_ax.grid()
        cur_ax.set_title("{}, R^2={}".format(t_name, round(regr.score(cur_X, cur_Y), 5)))
        print("")




    plt.tight_layout()
    plt.savefig("plots/example.pdf".format(t_name))

        
    

if __name__ == "__main__":
    main()




