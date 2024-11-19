# %%
import pickle
import numpy as np
import joblib

def load_inference_data(file_path, subsample=8):
    with open(file_path, 'rb') as f:
        all_voltage = pickle.load(f)
    
    X = all_voltage[:,1::subsample]
    indices = np.arange(X.shape[0])

    X = X.reshape(X.shape[0],1,X.shape[1])

    return X, indices

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

# Prepare data for predictions
X_new, indices = load_inference_data('all_voltage.pkl', subsample=8)

# Load the model
model = joblib.load('trained_model.pkl')

# %%
# Predict the first 2 elements of the dataset
y_pred = model.predict(X_new[0:2])
print(y_pred)

# %%
