import pytest
import numpy as np
import sys
sys.path.append('./')
from src.Lab_2_2_kNN import *
import matplotlib
matplotlib.use('Agg')

# Test Minkowski Distance
def test_minkowski_distance():
    a = np.array([0, 0])
    b = np.array([3, 4])

    # Test Euclidean distance (p=2)
    assert (
        minkowski_distance(a, b, p=2) == 5.0
    ), "Minkowski distance (Euclidean) is incorrect."

    # Test Manhattan distance (p=1)
    assert (
        minkowski_distance(a, b, p=1) == 7.0
    ), "Minkowski distance (Manhattan) is incorrect."

    # Test higher-order Minkowski distance
    assert pytest.approx(minkowski_distance(a, b, p=3), 1e-6) == (3**3 + 4**3) ** (
        1 / 3
    ), "Minkowski distance (p=3) is incorrect."

    # Test edge case: Zero distance
    assert (
        minkowski_distance(a, a) == 0.0
    ), "Minkowski distance for identical points is incorrect."


# Test kNN Class
@pytest.fixture
def knn_instance():
    model = knn()
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model.fit(X_train, y_train, k=3, p=2)
    return model


def test_knn_fit_invalid_inputs():
    model = knn()
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0])

    # Test mismatched input shapes
    with pytest.raises(
        ValueError, match="Length of X_train and y_train must be equal."
    ):
        model.fit(X_train, y_train)

    y_train = np.array([0, 1])
    # Test invalid k and p
    with pytest.raises(ValueError, match="k and p must be positive integers."):
        model.fit(X_train, y_train, k=-1, p=2)


def test_knn_predict(knn_instance):
    X_test = np.array([[4, 5], [1, 1]])
    predictions = knn_instance.predict(X_test)

    # Test predictions length
    assert len(predictions) == len(X_test), "Incorrect number of predictions."

    # Test predicted labels
    assert predictions[0] == 0, "Prediction for [4, 5] is incorrect."
    assert predictions[1] == 0, "Prediction for [1, 1] is incorrect."


def test_knn_predict_proba(knn_instance):
    X_test = np.array([[4, 5]])
    probabilities = knn_instance.predict_proba(X_test)

    # Test shape of probabilities
    assert probabilities.shape == (1, 2), "Incorrect shape for predicted probabilities."

    # Test sum of probabilities
    assert (
        pytest.approx(probabilities.sum(), 1e-6) == 1.0
    ), "Probabilities do not sum to 1."


def test_knn_compute_distances(knn_instance):
    distances = knn_instance.compute_distances(np.array([3, 3]))

    # Test length of distances
    assert len(distances) == len(
        knn_instance.x_train
    ), "Distance array length is incorrect."

    # Test a specific distance
    assert pytest.approx(distances[1], 1e-6) == 1.0, "Distance to [3, 4] is incorrect."


def test_knn_get_k_nearest_neighbors(knn_instance):
    distances = np.array([2.0, 1.0, 3.0, 0.5])
    neighbors = knn_instance.get_k_nearest_neighbors(distances)

    # Test number of neighbors
    assert len(neighbors) == knn_instance.k, "Incorrect number of nearest neighbors."

    # Test order of neighbors
    assert list(neighbors) == [3, 1, 0], "Nearest neighbors order is incorrect."


def test_knn_most_common_label(knn_instance):
    labels = np.array([1, 0, 1])
    most_common = knn_instance.most_common_label(labels)

    # Test majority label
    assert most_common == 1, "Majority label is incorrect."


def test_evaluate_classification_metrics():
    # Test case 1: Standard binary classification
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    positive_label = 1

    metrics = evaluate_classification_metrics(y_true, y_pred, positive_label)

    # Expected confusion matrix: [TN=2, FP=1, FN=1, TP=2]
    assert metrics["Confusion Matrix"] == [2, 1, 1, 2], "Confusion Matrix is incorrect."

    # Expected metrics
    assert metrics["Accuracy"] == pytest.approx(4 / 6, 1e-6), "Accuracy is incorrect."
    assert metrics["Precision"] == pytest.approx(2 / 3, 1e-6), "Precision is incorrect."
    assert metrics["Recall"] == pytest.approx(2 / 3, 1e-6), "Recall is incorrect."
    assert metrics["Specificity"] == pytest.approx(
        2 / 3, 1e-6
    ), "Specificity is incorrect."
    assert metrics["F1 Score"] == pytest.approx(2 / 3, 1e-6), "F1 Score is incorrect."

    # Test case 2: All predictions correct
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    metrics = evaluate_classification_metrics(y_true, y_pred, positive_label)

    # Expected confusion matrix: [TN=2, FP=0, FN=0, TP=2]
    assert metrics["Confusion Matrix"] == [
        2,
        0,
        0,
        2,
    ], "Confusion Matrix for perfect predictions is incorrect."

    # All metrics should be 1.0
    assert metrics["Accuracy"] == 1.0, "Accuracy for perfect predictions is incorrect."
    assert (
        metrics["Precision"] == 1.0
    ), "Precision for perfect predictions is incorrect."
    assert metrics["Recall"] == 1.0, "Recall for perfect predictions is incorrect."
    assert (
        metrics["Specificity"] == 1.0
    ), "Specificity for perfect predictions is incorrect."
    assert metrics["F1 Score"] == 1.0, "F1 Score for perfect predictions is incorrect."

    # Test case 3: No true positives or negatives
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    metrics = evaluate_classification_metrics(y_true, y_pred, positive_label)

    # Expected confusion matrix: [TN=0, FP=0, FN=4, TP=0]
    assert metrics["Confusion Matrix"] == [
        0,
        0,
        4,
        0,
    ], "Confusion Matrix for no positives is incorrect."

    # Metrics
    assert metrics["Accuracy"] == 0.0, "Accuracy for no positives is incorrect."
    assert metrics["Precision"] == 0.0, "Precision for no positives is incorrect."
    assert metrics["Recall"] == 0.0, "Recall for no positives is incorrect."
    assert metrics["Specificity"] == 0.0, "Specificity for no positives is incorrect."
    assert metrics["F1 Score"] == 0.0, "F1 Score for no positives is incorrect."

    # Test case 4: Edge case with no true labels
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    metrics = evaluate_classification_metrics(y_true, y_pred, positive_label)

    # Expected confusion matrix: [TN=0, FP=4, FN=0, TP=0]
    assert metrics["Confusion Matrix"] == [
        0,
        4,
        0,
        0,
    ], "Confusion Matrix for all negatives is incorrect."

    # Metrics
    assert metrics["Accuracy"] == 0.0, "Accuracy for all negatives is incorrect."
    assert metrics["Precision"] == 0.0, "Precision for all negatives is incorrect."
    assert metrics["Recall"] == 0.0, "Recall for all negatives is incorrect."
    assert metrics["Specificity"] == 0.0, "Specificity for all negatives is incorrect."
    assert metrics["F1 Score"] == 0.0, "F1 Score for all negatives is incorrect."


def test_plot_calibration_curve():
    # Test data
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1])
    positive_label = 1
    n_bins = 5

    # Call the function and capture the returned values
    result = plot_calibration_curve(y_true, y_probs, positive_label, n_bins)

    # Verify the returned keys
    assert (
        "bin_centers" in result and "true_proportions" in result
    ), "Function must return 'bin_centers' and 'true_proportions'."

    # Check lengths of bin_centers and true_proportions
    assert len(result["bin_centers"]) == n_bins, "Number of bins is incorrect."
    assert (
        len(result["true_proportions"]) == n_bins
    ), "Number of proportions is incorrect."

    # Check bin centers
    expected_bin_centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(
        result["bin_centers"], expected_bin_centers, atol=1e-6
    ), "Bin centers are incorrect."

    # Check true proportions
    expected_true_proportions = [
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]  # Computed manually for this test
    assert np.allclose(
        result["true_proportions"], expected_true_proportions, atol=1e-6
    ), "True proportions are incorrect."

    # Edge case: All probabilities are the same
    y_probs_uniform = np.array([0.5] * len(y_true))
    result_uniform = plot_calibration_curve(
        y_true, y_probs_uniform, positive_label, n_bins=1
    )

    assert result_uniform["true_proportions"][0] == np.mean(
        y_true == positive_label
    ), "True proportions for uniform probabilities are incorrect."


def test_plot_probability_histograms():
    # Test data
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1])
    positive_label = 1
    n_bins = 5

    # Call the function and capture returned data
    result = plot_probability_histograms(y_true, y_probs, positive_label, n_bins)

    # Verify returned structure
    assert (
        "array_passed_to_histogram_of_positive_class" in result
    ), "Return must contain 'array_passed_to_histogram_of_positive_class'."
    assert (
        "array_passed_to_histogram_of_negative_class" in result
    ), "Return must contain 'array_passed_to_histogram_of_negative_class'."

    # Validate arrays passed to histograms
    pos_probs = result["array_passed_to_histogram_of_positive_class"]
    neg_probs = result["array_passed_to_histogram_of_negative_class"]

    expected_pos_probs = np.array(
        [0.9, 0.8, 0.7, 0.6, 0.4]
    )  # Positive class probabilities
    expected_neg_probs = np.array(
        [0.1, 0.2, 0.3, 0.2, 0.1]
    )  # Negative class probabilities

    assert np.array_equal(
        np.sort(pos_probs), np.sort(expected_pos_probs)
    ), "Array of positive class probabilities is incorrect."
    assert np.array_equal(
        np.sort(neg_probs), np.sort(expected_neg_probs)
    ), "Array of negative class probabilities is incorrect."

    # Edge case: All predictions belong to one class
    y_true_all_positive = np.array([1, 1, 1, 1])
    y_probs_all_positive = np.array([0.5, 0.6, 0.7, 0.8])
    result_all_positive = plot_probability_histograms(
        y_true_all_positive, y_probs_all_positive, positive_label, n_bins
    )
    assert np.array_equal(
        result_all_positive["array_passed_to_histogram_of_positive_class"],
        y_probs_all_positive,
    ), "Positive class array is incorrect when all predictions are positive."
    assert (
        len(result_all_positive["array_passed_to_histogram_of_negative_class"]) == 0
    ), "Negative class array should be empty when no negatives are present."


def test_plot_roc_curve():
    # Test data
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    y_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1])
    positive_label = 1

    # Call the function and capture returned data
    result = plot_roc_curve(y_true, y_probs, positive_label)

    # Verify returned structure
    assert (
        "fpr" in result and "tpr" in result
    ), "Return must contain 'fpr' and 'tpr' keys."

    # Validate FPR and TPR lengths
    assert len(result["fpr"]) == len(
        result["tpr"]
    ), "FPR and TPR arrays must have the same length."
    assert len(result["fpr"]) == 11, "There should be 11 points for the thresholds."

    # Validate FPR and TPR values at key thresholds
    thresholds = np.linspace(0, 1, 11)
    for idx, thresh in enumerate(thresholds):
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        expected_tpr = tp / (tp + fn) if tp + fn != 0 else 0
        expected_fpr = fp / (fp + tn) if fp + tn != 0 else 0

        assert result["tpr"][idx] == pytest.approx(
            expected_tpr, 1e-6
        ), f"TPR at threshold {thresh} is incorrect."
        assert result["fpr"][idx] == pytest.approx(
            expected_fpr, 1e-6
        ), f"FPR at threshold {thresh} is incorrect."

    # Edge case: Perfect predictions
    y_true_perfect = np.array([1, 0, 1, 0])
    y_probs_perfect = np.array([1, 0, 1, 0])
    result_perfect = plot_roc_curve(y_true_perfect, y_probs_perfect, positive_label)

    assert np.array_equal(
        result_perfect["tpr"], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ), "TPR for perfect predictions is incorrect."
    assert np.array_equal(
        result_perfect["fpr"], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ), "FPR for perfect predictions is incorrect."

