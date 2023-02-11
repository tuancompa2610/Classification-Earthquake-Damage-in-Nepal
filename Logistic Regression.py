acc_baseline = y.value_counts(normalize = True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))

model_lr = make_pipeline(OneHotEncoder(use_cat_names = True), LogisticRegression(max_iter = 1000))
model_lr.fit(X_train, y_train)

lr_train_acc = accuracy_score(y_train, model_lr.predict(X_train))
lr_val_acc = accuracy_score(y_val, model_lr.predict(X_val))

print("Logistic Regression, Training Accuracy Score:", lr_train_acc)
print("Logistic Regression, Validation Accuracy Score:", lr_val_acc)
