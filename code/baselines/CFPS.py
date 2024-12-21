def CFPS(sources, target, LOC, topK=1, classifier='RandomForest', measureApp='AUC'):
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression

    # Step 1: Similarity scores
    M = len(sources)
    simiScores = np.zeros(M)
    NP = np.concatenate([np.mean(target[:, :-1], axis=0), np.std(target[:, :-1], axis=0)])
    for i, src in enumerate(sources):
        HP_i = np.concatenate([np.mean(src[:, :-1], axis=0), np.std(src[:, :-1], axis=0)])
        simiScores[i] = 1 / (1 + np.sqrt(np.sum((HP_i - NP) ** 2)))

    # Step 2: Applicability scores
    appScores = np.zeros((M, M))
    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'NaiveBayes': GaussianNB(),
        'Logistic': LogisticRegression(max_iter=1000)
    }
    for i in range(M):
        X_test, y_test = sources[i][:, :-1], sources[i][:, -1]
        for j in range(M):
            if i == j:
                continue
            X_train, y_train = sources[j][:, :-1], sources[j][:, -1]
            clf = classifiers[classifier]
            clf.fit(X_train, y_train)
            y_probs = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_probs)
            f1 = f1_score(y_test, clf.predict(X_test))
            appScores[i, j] = auc if measureApp == 'AUC' else f1

    # Step 3: Recommendation scores
    recScores = np.zeros(M)
    for i in range(M):
        for j in range(M):
            if i != j:
                recScores[i] += simiScores[j] * appScores[j, i]

    # Step 4: Training and prediction
    trainData = np.vstack([sources[idx] for idx in np.argsort(recScores)[-topK:]])
    X_train, y_train = trainData[:, :-1], trainData[:, -1]
    X_target, y_target = target[:, :-1], target[:, -1]

    clf = classifiers[classifier]
    clf.fit(X_train, y_train)
    y_probs = clf.predict_proba(X_target)[:, 1]

    PD = np.sum((y_probs > 0.5) & (y_target == 1)) / np.sum(y_target == 1)
    PF = np.sum((y_probs > 0.5) & (y_target == 0)) / np.sum(y_target == 0)
    auc = roc_auc_score(y_target, y_probs)
    f1 = f1_score(y_target, (y_probs > 0.5))

    return PD, PF, auc, f1
