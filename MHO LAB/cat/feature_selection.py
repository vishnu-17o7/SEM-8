from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class PSOFeatureSelector:
    def __init__(self, n_particles, max_iter, dim):
        self.n = n_particles
        self.iters = max_iter
        self.dim = dim

        self.positions = np.random.randint(2, size=(n_particles, dim))
        self.velocities = np.zeros((n_particles, dim))

        self.best_pos = None
        self.best_score = -np.inf

    def evaluate(self, mask):
        if mask.sum() == 0:
            return -np.inf
        

        X_train_evaluation = X_train.iloc[:, mask.astype(bool)]
        X_test_evaluation  = X_test.iloc[:, mask.astype(bool)]

        lr = LinearRegression()
        lr.fit(X_train_evaluation, y_train)
        pred = lr.predict(X_test_evaluation)
        return r2_score(y_test, pred)

    def update_velocity(self, p_best_pos):
        w = 0.7    
        c1 = 1.4 
        c2 = 1.4    

        r1 = np.random.rand(*self.velocities.shape)
        r2 = np.random.rand(*self.velocities.shape)

        self.velocities = (w * self.velocities +
                           c1 * r1 * (p_best_pos - self.positions) +
                           c2 * r2 * (self.best_pos   - self.positions))

    def update_position(self):
        self.positions = self.positions + self.velocities

    def fit(self, n_particles=30, max_iter=50):
        for _ in range(max_iter):
            for i in range(n_particles):
                score = self.evaluate(self.positions[i])
                if score > self.best_score:
                    print("Updating Best score...")
                    self.best_score = score
                    self.best_pos   = self.positions[i].copy()

            self.update_velocity(self.best_pos)
            self.update_position()
        
        return self.best_pos, self.best_score


print("Exploratory Data Analysis \n \n")

housing = fetch_california_housing(as_frame=True)
# print(housing)
data = housing.frame 
data = pd.DataFrame(data = data)
print("First 5 rows of data:  ")
print(data.head())

print("\n\nShape of data: ")
print(data.shape)

print("\n\nStat of data: ")
print(data.describe())

print("\n\nColumn Data Types: ")
print(data.dtypes)

print("\n\nMissing values per column:")
print(data.isna().sum())



X = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']

scaler = MinMaxScaler()
# # X['MedInc'] = scaler.fit_transform(X['MedInc'])
# y = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n\nTraining Linear Regression model with all features")

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
non_pso_accuracy = r2_score(y_test,y_pred)
print("accuracy score:",non_pso_accuracy)

pso_feature,pso_feature_accuracy = PSOFeatureSelector(n_particles=1000,max_iter=10000,dim=X.shape[1]).fit()

print("\nBest particle pos:", pso_feature)
print("Selected feature indices:",
      np.where(pso_feature == 1)[0])

selected_features =  np.where(pso_feature == 1)[0]
index = 0

print("Selected Column names: ")
for name in data.columns:
    if(index in selected_features):
        print(name)
    index = index + 1
    
print("The accuracy after PSO feature selection:",pso_feature_accuracy)

if(non_pso_accuracy < pso_feature_accuracy):
    print("The pso selected features were better in terms of accuracy")
else:
    print("The normal regression model was better in terms of accuracy")


# pso_x_train, pso_x_test, pso_y_train, pso_y_test = train_test_split(X.loc[:, np.where(feature == 1)[0]], y,test_size=0.2, random_state=42)

# model = LinearRegression()


# model.fit(pso_x_train, pso_y_train)

# pso_y_pred = model.predict(pso_x_test)
# print("accuracy score:", r2_score(pso_y_test,pso_y_pred))

