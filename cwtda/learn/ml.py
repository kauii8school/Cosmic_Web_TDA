import sklearn.model_selection 
import sklearn.metrics 
import sklearn.svm
import keras.models 
import keras.layers

def svr(pl_lst, show_score=False):
    y = [i[0] for i in pl_lst]
    X = [np.stack(i[1]).ravel() for i in pl_lst]

    model = sklearn.svm.SVR(C=1.0)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if show_score:
        print(f"Explained variance Score: {sklearn.metrics.explained_variance_score(y_test, y_pred)}")
        print(f"Max Error: {sklearn.metrics.max_error(y_test, y_pred)}")
        print(f"Mean Absolute Error: {sklearn.metrics.mean_absolute_error(y_test, y_pred)}")
        print(f"Mean Squared Error: {sklearn.metrics.mean_squared_error(y_test, y_pred)}")
        print(f"Mean Squared Log Error : {sklearn.metrics.mean_squared_log_error(y_test, y_pred)}")
        print(f"R2 Score : {sklearn.metrics.r2_score(y_test, y_pred)}")


    return model 

def nn(pl_lst, show_score=False):
    y = np.array([i[0] for i in pl_lst]).reshape(-1, 1)
    X = np.array([np.stack(i[1]).ravel() for i in pl_lst])

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(400, input_dim=int(X[0].shape[0]), activation='relu'))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=.8)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

    print(model.summary)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if show_score:
        print(f"Explained variance Score: {sklearn.metrics.explained_variance_score(y_test, y_pred)}")
        print(f"Max Error: {sklearn.metrics.max_error(y_test, y_pred)}")
        print(f"Mean Absolute Error: {sklearn.metrics.mean_absolute_error(y_test, y_pred)}")
        print(f"Mean Squared Error: {sklearn.metrics.mean_squared_error(y_test, y_pred)}")
        print(f"Mean Squared Log Error : {sklearn.metrics.mean_squared_log_error(y_test, y_pred)}")
        print(f"R2 Score : {sklearn.metrics.r2_score(y_test, y_pred)}")

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    return model 