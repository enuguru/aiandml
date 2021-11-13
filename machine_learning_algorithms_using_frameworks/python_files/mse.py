from sklearn.metrics import mean_squared_error
y_true = [3,5,2]
y_pred = [3,4,4]
print(mean_squared_error(y_true, y_pred, squared=False))
