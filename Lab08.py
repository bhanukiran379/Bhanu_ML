import numpy as np
import matplotlib.pyplot as plt

CONVERGENCE_ERROR = 0.002
MAX_EPOCHS = 1000

def summation_unit(inputs, weights):

    return float(np.dot(inputs, weights))

def step_activation(y):

    return 1.0 if y > 0 else 0.0

def bipolar_step_activation(y):
  
    if y > 0:
        return 1.0
    elif y < 0:
        return -1.0
    else:
        return 0.0


def sigmoid_activation(y):

    return 1.0 / (1.0 + np.exp(-y))


def tanh_activation(y):
   
    return np.tanh(y)


def relu_activation(y):
   
    return y if y > 0 else 0.0


def leaky_relu_activation(y, alpha=0.01):
  
    return y if y > 0 else alpha * y


def comparator_unit(target, output):
 
    return target - output


def sum_square_error(targets, outputs):
    
    targets = np.array(targets, dtype=float)
    outputs = np.array(outputs, dtype=float)
    return float(np.sum((targets - outputs) ** 2))

def get_activation_function(name):
   
    mapping = {
        "step": step_activation,
        "bipolar_step": bipolar_step_activation,
        "sigmoid": sigmoid_activation,
        "tanh": tanh_activation,
        "relu": relu_activation,
        "leaky_relu": leaky_relu_activation,
    }
    return mapping[name]


def train_perceptron(X, T, weights_init, learning_rate, activation_name,
                      convergence_error=CONVERGENCE_ERROR, max_epochs=MAX_EPOCHS):
  
    activation_fn = get_activation_function(activation_name)

    # Prepend bias input (x0 = 1) to every sample
    bias_col = np.ones((X.shape[0], 1))
    X_bias = np.hstack([bias_col, X])

    weights = np.array(weights_init, dtype=float).copy()
    epoch_errors = []
    epochs_to_converge = max_epochs

    for epoch in range(1, max_epochs + 1):
        outputs = []
        for i in range(X_bias.shape[0]):
            y = summation_unit(X_bias[i], weights)
            o = activation_fn(y)
            e = comparator_unit(T[i], o)
            weights = weights + learning_rate * e * X_bias[i]
            outputs.append(o)

        epoch_sse = sum_square_error(T, outputs)
        epoch_errors.append(epoch_sse)

        if epoch_sse <= convergence_error:
            epochs_to_converge = epoch
            break

    return {
        "weights": weights,
        "epoch_errors": epoch_errors,
        "epochs_to_converge": epochs_to_converge,
    }


def plot_epoch_vs_error(epoch_errors, title, save_path):
    """Plots epoch number (x) vs sum-square-error (y) and saves to file."""
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(epoch_errors) + 1), epoch_errors, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Sum-Square-Error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_iterations_vs_learning_rate(learning_rates, iterations, title, save_path):
    """Plots learning rate (x) vs epochs-to-converge (y) and saves to file."""
    plt.figure(figsize=(6, 4))
    plt.plot(learning_rates, iterations, marker="s")
    plt.xlabel("Learning Rate")
    plt.ylabel("Iterations to Converge")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===========================================================================
# A6 / A7. Customer transaction perceptron + pseudo-inverse comparison
# ===========================================================================
def normalize_features(X):
    """Min-max normalizes each column of X to the [0, 1] range."""
    X = np.array(X, dtype=float)
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom = np.where((col_max - col_min) == 0, 1, col_max - col_min)
    return (X - col_min) / denom


def pseudo_inverse_solution(X, T):
    """
    Solves for weights using the Moore-Penrose pseudo-inverse:
        W = (X^T X)^-1 X^T T    (computed via np.linalg.pinv)
    X : ndarray with bias column already included, shape (n_samples, n_features+1)
    T : ndarray of targets, shape (n_samples,)
    returns : weight vector W
    """
    return np.linalg.pinv(X) @ T


def predict_with_weights(X, weights, activation_name):
    """Generates predictions for a bias-augmented input matrix X."""
    activation_fn = get_activation_function(activation_name)
    preds = []
    for i in range(X.shape[0]):
        y = summation_unit(X[i], weights)
        preds.append(activation_fn(y))
    return np.array(preds)


# ===========================================================================
# A8 / A9 / A10. Back-propagation for a 2-2-1 (and 2-2-2) MLP
# ===========================================================================
def sigmoid_derivative_from_output(o):
    """Given sigmoid output o, returns o*(1-o) - the local derivative."""
    return o * (1.0 - o)


def initialize_backprop_weights(n_in, n_hidden, n_out, seed=42):
    """Initializes small random weights for a single-hidden-layer MLP."""
    rng = np.random.default_rng(seed)
    V = rng.uniform(-0.05, 0.05, size=(n_in, n_hidden))   # input -> hidden
    W = rng.uniform(-0.05, 0.05, size=(n_hidden, n_out))  # hidden -> output
    b_hidden = rng.uniform(-0.05, 0.05, size=(n_hidden,))
    b_output = rng.uniform(-0.05, 0.05, size=(n_out,))
    return {"V": V, "W": W, "b_hidden": b_hidden, "b_output": b_output}


def forward_pass(x, params):
    """Forward-propagates one sample x through the hidden and output layers."""
    net_h = x @ params["V"] + params["b_hidden"]
    o_h = sigmoid_activation(net_h) if np.isscalar(net_h) else 1.0 / (1.0 + np.exp(-net_h))
    net_o = o_h @ params["W"] + params["b_output"]
    o_o = 1.0 / (1.0 + np.exp(-net_o))
    return o_h, o_o


def backward_pass_update(x, t, o_h, o_o, params, learning_rate):
    """
    Applies one step of the back-propagation weight update (T4.3-T4.5)
    for a single training sample.
    """
    # Output layer error term: delta_k = o_k(1-o_k)(t_k - o_k)
    delta_k = sigmoid_derivative_from_output(o_o) * (t - o_o)

    # Hidden layer error term: delta_h = o_h(1-o_h) * sum_k(w_kh * delta_k)
    delta_h = sigmoid_derivative_from_output(o_h) * (params["W"] @ delta_k)

    # Weight updates: delta_w_ji = eta * delta_j * x_ji
    params["W"] += learning_rate * np.outer(o_h, delta_k)
    params["b_output"] += learning_rate * delta_k

    params["V"] += learning_rate * np.outer(x, delta_h)
    params["b_hidden"] += learning_rate * delta_h

    return params


def train_backprop_mlp(X, T, n_hidden, learning_rate,
                        convergence_error=CONVERGENCE_ERROR, max_epochs=MAX_EPOCHS, seed=42):
    """
    Trains a single-hidden-layer MLP with the back-propagation algorithm.
    X : ndarray shape (n_samples, n_in)
    T : ndarray shape (n_samples, n_out)   (n_out can be 1 or 2)
    returns : dict with trained params, epoch-wise SSE list, epochs_to_converge
    """
    n_in = X.shape[1]
    n_out = T.shape[1]
    params = initialize_backprop_weights(n_in, n_hidden, n_out, seed=seed)

    epoch_errors = []
    epochs_to_converge = max_epochs

    for epoch in range(1, max_epochs + 1):
        outputs = []
        for i in range(X.shape[0]):
            o_h, o_o = forward_pass(X[i], params)
            params = backward_pass_update(X[i], T[i], o_h, o_o, params, learning_rate)
            outputs.append(o_o)

        outputs = np.array(outputs)
        epoch_sse = float(np.sum((T - outputs) ** 2))
        epoch_errors.append(epoch_sse)

        if epoch_sse <= convergence_error:
            epochs_to_converge = epoch
            break

    return {
        "params": params,
        "epoch_errors": epoch_errors,
        "epochs_to_converge": epochs_to_converge,
    }


# ===========================================================================
# A11 / A12. Scikit-learn MLPClassifier wrappers
# ===========================================================================
def train_sklearn_mlp(X, y, hidden_layer_sizes=(2,), activation="logistic",
                       learning_rate_init=0.05, max_iter=1000, seed=42):
    """Trains sklearn's MLPClassifier and returns the fitted model."""
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                           activation=activation,
                           learning_rate_init=learning_rate_init,
                           max_iter=max_iter,
                           random_state=seed)
    model.fit(X, y)
    return model


def build_tfidf_author_dataset(excel_path, max_features=300, min_df=2):
    """
    Loads the project dataset (blog paragraphs) and builds a
    TF-IDF feature matrix with 'Author' as the classification target.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_excel(excel_path)
    df = df.dropna(subset=["Author", "Blog"])

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df,
                                  stop_words="english")
    X = vectorizer.fit_transform(df["Blog"]).toarray()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Author"])

    return X, y, label_encoder, vectorizer


# ===========================================================================
# MAIN PROGRAM
# ===========================================================================
if __name__ == "__main__":

    # -----------------------------------------------------------------
    # Common data: AND gate and XOR gate truth tables
    # -----------------------------------------------------------------
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    T_and = np.array([0, 0, 0, 1])

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    T_xor = np.array([0, 1, 1, 0])

    initial_weights = [10, 0.2, -0.75]   # W0, W1, W2
    LEARNING_RATE = 0.05

    print("=" * 70)
    print("A2. Perceptron - Step activation - AND Gate")
    print("=" * 70)
    result_a2 = train_perceptron(X_and, T_and, initial_weights, LEARNING_RATE, "step")
    print(f"Epochs to converge : {result_a2['epochs_to_converge']}")
    print(f"Final weights      : {result_a2['weights']}")
    print(f"Final epoch SSE    : {result_a2['epoch_errors'][-1]:.5f}")
    plot_epoch_vs_error(result_a2["epoch_errors"],
                         "A2: AND Gate (Step) - Epoch vs SSE",
                         "/A2_and_step_epoch_vs_error.png")

    print()
    print("=" * 70)
    print("A3. AND Gate - Compare activation functions")
    print("=" * 70)
    a3_results = {}
    for act_name in ["bipolar_step", "sigmoid", "relu"]:
        res = train_perceptron(X_and, T_and, initial_weights, LEARNING_RATE, act_name)
        a3_results[act_name] = res
        print(f"{act_name:15s} -> epochs to converge = {res['epochs_to_converge']}")
    print("(comparison also includes A2's 'step' result above)")

    print()
    print("=" * 70)
    print("A4. AND Gate (Step) - Vary learning rate")
    print("=" * 70)
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    iterations_list = []
    for lr in learning_rates:
        res = train_perceptron(X_and, T_and, initial_weights, lr, "step")
        iterations_list.append(res["epochs_to_converge"])
        print(f"lr = {lr:.1f} -> epochs to converge = {res['epochs_to_converge']}")
    plot_iterations_vs_learning_rate(learning_rates, iterations_list,
                                      "A4: AND Gate (Step) - LR vs Iterations",
                                      "/A4_and_step_lr_vs_iterations.png")

    print()
    print("=" * 70)
    print("A5. XOR Gate - Repeat A1 to A3")
    print("=" * 70)
    for act_name in ["step", "bipolar_step", "sigmoid", "relu"]:
        res = train_perceptron(X_xor, T_xor, initial_weights, LEARNING_RATE, act_name)
        print(f"XOR / {act_name:15s} -> epochs to converge = {res['epochs_to_converge']} "
              f"(1000 => did NOT converge, as expected for a single-layer perceptron)")
    res_xor_step = train_perceptron(X_xor, T_xor, initial_weights, LEARNING_RATE, "step")
    plot_epoch_vs_error(res_xor_step["epoch_errors"],
                         "A5: XOR Gate (Step) - Epoch vs SSE",
                         "/A5_xor_step_epoch_vs_error.png")

    print()
    print("=" * 70)
    print("A6 / A7. Customer transactions - Sigmoid Perceptron vs Pseudo-inverse")
    print("=" * 70)
    X_cust_raw = np.array([
        [20, 6, 2], [16, 3, 6], [27, 6, 2], [19, 1, 2], [24, 4, 2],
        [22, 1, 5], [15, 4, 2], [18, 4, 2], [21, 1, 4], [16, 2, 4],
    ], dtype=float)
    # Payment column included as a 4th feature
    payment = np.array([386, 289, 393, 110, 280, 167, 271, 274, 148, 198], dtype=float)
    X_cust_raw = np.hstack([X_cust_raw, payment.reshape(-1, 1)])
    T_cust = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=float)  # Yes=1, No=0

    X_cust_norm = normalize_features(X_cust_raw)
    cust_init_weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # W0..W4, chosen as zeros
    CUST_LR = 0.05

    res_cust = train_perceptron(X_cust_norm, T_cust, cust_init_weights, CUST_LR, "sigmoid")
    print(f"A6 Perceptron(sigmoid) -> epochs to converge = {res_cust['epochs_to_converge']}")
    print(f"A6 Final weights                              = {res_cust['weights']}")

    bias_col = np.ones((X_cust_norm.shape[0], 1))
    X_cust_bias = np.hstack([bias_col, X_cust_norm])
    pinv_weights = pseudo_inverse_solution(X_cust_bias, T_cust)
    pinv_preds = predict_with_weights(X_cust_bias, pinv_weights, "step")
    print(f"A7 Pseudo-inverse weights                     = {pinv_weights}")
    print(f"A7 Pseudo-inverse predictions (thresholded)   = {pinv_preds}")
    print(f"A7 Actual targets                             = {T_cust}")

    print()
    print("=" * 70)
    print("A8. Back-propagation (2-2-1 MLP) - AND Gate")
    print("=" * 70)
    T_and_col = T_and.reshape(-1, 1).astype(float)
    res_a8 = train_backprop_mlp(X_and.astype(float), T_and_col, n_hidden=2,
                                 learning_rate=0.05)
    print(f"Epochs to converge : {res_a8['epochs_to_converge']}")
    print(f"Final epoch SSE    : {res_a8['epoch_errors'][-1]:.5f}")
    plot_epoch_vs_error(res_a8["epoch_errors"],
                         "A8: AND Gate Backprop (2-2-1) - Epoch vs SSE",
                         "/A8_and_backprop_epoch_vs_error.png")

    print()
    print("=" * 70)
    print("A9. Back-propagation (2-2-1 MLP) - XOR Gate")
    print("=" * 70)
    T_xor_col = T_xor.reshape(-1, 1).astype(float)
    res_a9 = train_backprop_mlp(X_xor.astype(float), T_xor_col, n_hidden=2,
                                 learning_rate=0.05)
    print(f"Epochs to converge : {res_a9['epochs_to_converge']}")
    print(f"Final epoch SSE    : {res_a9['epoch_errors'][-1]:.5f}")
    plot_epoch_vs_error(res_a9["epoch_errors"],
                         "A9: XOR Gate Backprop (2-2-1) - Epoch vs SSE",
                         "/A9_xor_backprop_epoch_vs_error.png")

    print()
    print("=" * 70)
    print("A10. Back-propagation with 2 output nodes - AND & XOR")
    print("=" * 70)
    # 0 -> [1, 0] ; 1 -> [0, 1]
    T_and_2out = np.array([[1, 0] if t == 0 else [0, 1] for t in T_and], dtype=float)
    T_xor_2out = np.array([[1, 0] if t == 0 else [0, 1] for t in T_xor], dtype=float)

    res_a10_and = train_backprop_mlp(X_and.astype(float), T_and_2out, n_hidden=2, learning_rate=0.05)
    res_a10_xor = train_backprop_mlp(X_xor.astype(float), T_xor_2out, n_hidden=2, learning_rate=0.05)
    print(f"AND (2-output) -> epochs to converge = {res_a10_and['epochs_to_converge']}")
    print(f"XOR (2-output) -> epochs to converge = {res_a10_xor['epochs_to_converge']}")

    print()
    print("=" * 70)
    print("A11. sklearn MLPClassifier - AND & XOR")
    print("=" * 70)
    model_and = train_sklearn_mlp(X_and, T_and, hidden_layer_sizes=(2,))
    model_xor = train_sklearn_mlp(X_xor, T_xor, hidden_layer_sizes=(2,))
    print(f"AND predictions : {model_and.predict(X_and)}  (actual: {T_and})")
    print(f"XOR predictions : {model_xor.predict(X_xor)}  (actual: {T_xor})")

    print()
    print("=" * 70)
    print("A12. sklearn MLPClassifier - Project dataset (Author classification)")
    print("=" * 70)
    X_proj, y_proj, label_enc, vectorizer = build_tfidf_author_dataset(
        "splt_dataset.xlsx")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X_proj, y_proj, test_size=0.2, random_state=42, stratify=y_proj)

    model_proj = train_sklearn_mlp(X_train, y_train, hidden_layer_sizes=(50,),
                                    activation="relu", learning_rate_init=0.01,
                                    max_iter=500)
    y_pred = model_proj.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Number of authors (classes) : {len(label_enc.classes_)}")
    print(f"Train samples / Test samples: {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"Test accuracy                : {acc:.4f}")

    print()
    print("All plots have been saved to this folder")
