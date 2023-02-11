# Build Model
model = make_pipeline(OrdinalEncoder(), DecisionTreeClassifier(random_state = 42))
# Fit model to training data
model.fit(X_train, y_train)

acc_train = accuracy_score(y_train, model.predict(X_train))
acc_val = accuracy_score(y_val, model.predict(X_val))

print("Decision Tree, Training Accuracy:", round(acc_train, 2))
print("Decision Tree, Validation Accuracy:", round(acc_val, 2))

tree_depth = model.named_steps["decisiontreeclassifier"].get_depth()
print("Tree Depth:", tree_depth)

depth_hyperparams = range(1, 70, 2)
training_acc = []
validation_acc = []

for d in depth_hyperparams:
    # Create model with `max_depth` of `d`
    test_model = make_pipeline(OrdinalEncoder(), DecisionTreeClassifier(max_depth = d, random_state = 42))
    # Fit model to training data
    test_model.fit(X_train, y_train)
    # Calculate training accuracy score and append to `training_acc`
    training_acc.append(accuracy_score(y_train, test_model.predict(X_train)))
    # Calculate validation accuracy score and append to `training_acc`
    validation_acc.append(accuracy_score(y_val, test_model.predict(X_val)))

print("Training Accuracy Scores:", training_acc[:3])
print("Validation Accuracy Scores:", validation_acc[:3])

# Plot `depth_hyperparams`, `training_acc`
plt.plot(depth_hyperparams, training_acc, label = "training")
plt.plot(depth_hyperparams, validation_acc, label = "validation")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.legend()

retrain_model = make_pipeline(OrdinalEncoder(), DecisionTreeClassifier(max_depth = 7, random_state = 42))
retrain_model.fit(X_train, y_train)
test_acc = accuracy_score(y_test, retrain_model.predict(X_test))
print("Test Accuracy:", round(test_acc, 2))
