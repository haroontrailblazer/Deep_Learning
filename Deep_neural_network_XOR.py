from sklearn.neural_network import MLPClassifier
import numpy as np

x=np.array([[0,0],[1,1],[1,0],[0,1]])
y=np.array([0,0,1,1])

model = MLPClassifier(hidden_layer_sizes=(4, 4),activation='tanh',max_iter=1000)
model.fit(x,y)

# Test the model
print(model.predict(x))

# plotting iteration and cost
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.plot(model.loss_curve_);plt.title('Iteration vs Cost');plt.xlabel('Iteration');plt.ylabel('Cost');plt.show()

# plotting itertion vs accuracy
accuracy = model.score(x, y)
plt.bar(['Final Accuracy'], [accuracy]);plt.title('Accuracy');plt.ylabel('Accuracy');plt.show()