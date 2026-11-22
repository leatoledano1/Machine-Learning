from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import *
### HELPER FUNCTIONS ###
# Add here any helper functions which you think will be useful


### Exercise Solution ###

if __name__ == "__main__":
    n_samples = 200
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.2, random_state=0)

    X_circles, y_circles = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=0)
    cov = [[0.5, 0.2], [0.2, 0.5]]
    X1 = np.random.multivariate_normal([-1, -1], cov, n_samples // 2)
    X2 = np.random.multivariate_normal([1, 1], cov, n_samples // 2)
    X_gauss = np.vstack([X1, X2])
    y_gauss = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    datasets = [
        ("Moons", X_moons, y_moons),
        ("Circles", X_circles, y_circles),
        ("Gaussians", X_gauss, y_gauss)
    ]

    classifiers = [
        ("SVM (C=5)", SVC(C=5, kernel='rbf')),
        ("Decision Tree (Depth 7)", DecisionTreeClassifier(max_depth=7)),
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5))
    ]

    fig = make_subplots(rows=3, cols=3,
                        subplot_titles=[f"{d[0]} - {c[0]}" for d in datasets for c in classifiers],
                        horizontal_spacing=0.05, vertical_spacing=0.07)

    for i, (ds_name, X, y) in enumerate(datasets):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for j, (clf_name, clf) in enumerate(classifiers):

            clf.fit(X_train, y_train)

            acc = accuracy_score(y_test, clf.predict(X_test))

            x_range = [X[:, 0].min() - .5, X[:, 0].max() + .5]
            y_range = [X[:, 1].min() - .5, X[:, 1].max() + .5]

            fig.add_trace(decision_surface(clf.predict, x_range, y_range, showscale=False),
                          row=i + 1, col=j + 1)

            fig.add_trace(go.Scatter(x=X_test[:, 0], y=X_test[:, 1], mode='markers',
                                     marker=dict(color=y_test, colorscale=custom,
                                    line=dict(width=1, color='black')), showlegend=False), row=i + 1, col=j + 1)

            fig.layout.annotations[i * 3 + j].update(text=f"{ds_name} | {clf_name}<br>Accuracy: {acc:.2f}")

    fig.update_layout(height=1000, width=1000, title_text="Comparison of Classification Decision Boundaries",
                      title_y=0.98,margin=dict(t=100))
    fig.show()



