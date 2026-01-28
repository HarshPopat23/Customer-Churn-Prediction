from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )
    log_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    return (
        log_model,
        rf_model,
        X_train, X_test,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler
    )
