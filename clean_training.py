# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from detach_rocket.detach_classes import DetachRocket
from sklearn.model_selection import train_test_split
import joblib

# Silence warnings
import warnings
warnings.filterwarnings("ignore")

def load_training_data(file_path, subsample=8, test_size=0.2, random_state=42):

    with open(file_path, 'rb') as f:
        all_voltage = pickle.load(f)

    X = all_voltage[:,1::subsample]
    y = all_voltage[:,0]

    # Assign an index to each instance
    indices = np.arange(len(y))

    # transform labels 2 and 3 to 0
    y_bin = np.where(y == 2, 0, y)
    y_bin = np.where(y_bin == 3, 0, y_bin)

    (X_train, X_test, y_train, y_test, indices_train, indices_test) = train_test_split(X, y_bin, indices, test_size=test_size, stratify=y_bin, random_state=random_state)

    X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

    indices_train = indices_train
    indices_test = indices_test

    return X_train, X_test, y_train, y_test, indices_train, indices_test


class VoltageClassifier:
    def __init__(self, X_train, X_test, y_train, y_test, random_state=42, model_type="multirocket", num_kernels=5000, trade_off=0.05):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state
        self.model_type = model_type
        self.num_kernels = num_kernels
        self.trade_off = trade_off

    def train_model(self):
        np.random.seed(self.random_state)
        self.DetachRocketModel = DetachRocket(model_type=self.model_type, num_kernels = self.num_kernels, trade_off=self.trade_off)
        self.DetachRocketModel.fit(self.X_train,self.y_train)

    def predict(self, X_new):
        return self.DetachRocketModel.predict(X_new)

    def evaluate_model(self):
        detach_test_score, _ = self.DetachRocketModel.score(self.X_test,self.y_test)
        print('Test Accuraccy: {:.2f}%'.format(100*detach_test_score))

    def detach_curve_plot(self):
        percentage_vector = self.DetachRocketModel._percentage_vector
        acc_curve = self.DetachRocketModel._sfd_curve

        c = self.DetachRocketModel.trade_off

        x_plot=(percentage_vector) * 100
        y_plot=(acc_curve/acc_curve[0]-1) * 100

        point_x = x_plot[self.DetachRocketModel._max_index]

        plt.figure(figsize=(8,3.5))
        plt.axvline(x = point_x, color = 'r',label=f'Optimal Model (c={c})')
        plt.plot(x_plot, y_plot, label='SFD curve', linewidth=2.5, color='C7', alpha=1)
        plt.grid(True, linestyle='-', alpha=0.5)
        plt.xlim(102,-2)
        plt.xlabel('% of Retained Features')
        plt.ylabel('Relative Validation Set Accuracy (%)')
        plt.legend()
        plt.show()

        print('Optimal Model Size: {:.2f}% of full model'.format(point_x))

# %%
## Train the model

random_state = 42
subsample = 8

X_train, X_test, y_train, y_test, indices_train, indices_test = load_training_data('all_voltage.pkl', random_state=random_state, subsample=subsample)

model = VoltageClassifier(X_train, X_test, y_train, y_test, random_state=random_state)
model.train_model()

# Evaluate the model
model.evaluate_model()

# Plot the detach curve
model.detach_curve_plot()

# Save the model
joblib.dump(model, 'trained_model.pkl')


# %%
