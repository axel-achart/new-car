import numpy as np
import pandas as pd


### A l’aide de la librairie Numpy, creez VOTRE Class LinearRegression.
### N’utilisez pas de fonctions de regression lineaire existante.


# --------------- PREPARATION DES DONNEES ---------------
def load_and_prepare_data(csv_path, reference_year=2017):
    # 1) Lecture du dataset
    df = pd.read_csv(csv_path)

    # 2) Nettoyage basique
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # 3) Creation de la variable Age (consigne: annee de reference = 2017)
    df["Age"] = reference_year - df["Year"]

    # 4) Selection des variables pour un modele univarie
    # X = feature(s), y = target
    X = df[["Age"]].to_numpy()
    y = df["Selling_Price"].to_numpy()

    return df, X, y


def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    # Melange reproductible des indices
    rng = np.random.default_rng(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    # Decoupe train/test
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    # Extraction des sous-ensembles
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test


# --------------- CLASSE LINEAR REGRESSION (NUMPY) ---------------
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # learning_rate = taille du pas de mise a jour
        # n_iterations = nombre de passages de la descente de gradient
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # X: matrice (n_samples, n_features)
        # y: vecteur (n_samples,)
        n_samples, n_features = X.shape

        # Initialisation des parametres
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Boucle d'apprentissage
        for _ in range(self.n_iterations):
            # 1) Prediction
            y_pred = np.dot(X, self.weights) + self.bias

            # 2) Calcul des gradients (derivees de la MSE)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 3) Mise a jour des parametres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        # Prediction lineaire
        return np.dot(X, self.weights) + self.bias


# --------------- ETAPE 1 : ENTRAINEMENT ---------------
CSV_PATH = "data/raw/carData.csv"

df, X, y = load_and_prepare_data(CSV_PATH, reference_year=2017)
X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2, random_state=42)

print(df[["Year", "Age", "Selling_Price"]].head())
print(f"Dataset shape apres nettoyage: {df.shape}")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

model = LinearRegression(learning_rate=0.01, n_iterations=5000)
model.fit(X_train, y_train)

print("\n=== Etape 1 : Entrainement termine ===")
print(f"Poids appris (w): {model.weights}")
print(f"Biais appris (b): {model.bias:.4f}")


# --------------- ETAPE 2 : PREDICTION ---------------
y_pred = model.predict(X_test)

print("\n=== Etape 2 : Prediction sur X_test ===")
print("Exemple (5 premieres predictions):")
for i in range(5):
    print(f"y_true={y_test[i]:.4f} | y_pred={y_pred[i]:.4f}")


# --------------- ETAPE 3 : METRIQUES (NUMPY ONLY) ---------------
def mean_absolute_error_numpy(y_true, y_pred):
    # MAE = moyenne des erreurs absolues
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error_numpy(y_true, y_pred):
    # MSE = moyenne des erreurs au carre
    return np.mean((y_true - y_pred) ** 2)


def rmse_numpy(y_true, y_pred):
    # RMSE = racine carree du MSE
    return np.sqrt(mean_squared_error_numpy(y_true, y_pred))


def r2_score_numpy(y_true, y_pred):
    # R^2 = 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


mae = mean_absolute_error_numpy(y_test, y_pred)
mse = mean_squared_error_numpy(y_test, y_pred)
rmse = rmse_numpy(y_test, y_pred)
r2 = r2_score_numpy(y_test, y_pred)

print("\n=== Etape 3 : Evaluation du modele (Numpy) ===")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")