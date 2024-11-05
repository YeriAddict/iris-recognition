from iris_recognition import DataLoader, IrisRecognitionModel

def main():
    # Tunable parameters
    kernel_size = 31
    f = 0.075
    thresholds = [0.140, 0.150, 0.155, 0.160, 0.165, 0.170, 0.180, 0.190, 0.200, 0.300, 0.400]

    # Load the training and testing data
    training, testing = DataLoader.create().load()

    # Create the Iris Recognition Model using the tuned parameters
    iris_model = IrisRecognitionModel(training, testing, kernel_size, f, thresholds)

    # Extract features and labels from the training and testing data
    X_train, X_test, y_train, y_test = iris_model.extract_features_and_labels()

    # Fit the model using the training data
    iris_model.fit(X_train, y_train)

    # Identification mode: Identify the class labels for the testing data
    y_pred = iris_model.identify(X_test)

    # Verification mode: Verify the testing data against the true labels
    y_verif = iris_model.verify(X_test, y_test)

    # Evaluate the Correct Recognition Rate (CRR) and False Match Rate (FMR) and False Non-Match Rate (FNMR)
    crr = iris_model.evaluate_crr(y_test, y_pred)
    fmr, fnmr = iris_model.evaluate_fmr_fnmr(y_verif)

    # Print the results
    print(f"{'L1 distance measure':<25} | {round(crr['L1'], 4):>6} %")
    print(f"{'L2 distance measure':<25} | {round(crr['L2'], 4):>6} %")
    print(f"{'Cosine distance measure':<25} | {round(crr['COSINE'], 4):>6} %")

    fmr_rounded = {k: round(v, 2) for k, v in fmr['COSINE'].items()}
    print("Cosine FMR |")
    for threshold, rate in fmr_rounded.items():
        print(f"threshold {threshold:<5}: {rate:>6} %")
    
    fnmr_rounded = {k: round(v, 2) for k, v in fnmr['COSINE'].items()}
    print("Cosine FNMR |")
    for threshold, rate in fnmr_rounded.items():
        print(f"threshold {threshold:<5}: {rate:>6} %")

    # Plot the ROC curve
    iris_model.plot_det_curve(fmr, fnmr)

if __name__ == "__main__":
    main()