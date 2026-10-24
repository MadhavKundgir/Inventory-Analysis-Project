import joblib 
from pathlib import Path

from data_preprocessing import load_vendor_invoice_data,prepare_features,split_data
from model_evaluation import (train_linear_regression,train_decision_tree,train_rondom_forest,evaluate_model)

def main():
    db_path = "C:/Users/Madhav/Inventory Analysis Project/data/inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # load_data
    df = load_vendor_invoice_data(db_path)

    # preper data
    X, y = prepare_features(df)
    X_train,X_test,y_train,y_test = split_data(X, y)

    # train_models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_rondom_forest(X_train, y_train)

    # Evaluate model 
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree Regressor"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regressor"))

    # select best model (lowest MAE)
    best_model_info = min(results, key = lambda x:x['mae'])
    best_model_name = best_model_info['model_name']

    best_model = {
        "Linear Regression" : lr_model,
        "Decision Tree Regressor" : dt_model,
        "Random Forest Regressor" : rf_model
    }[best_model_name]

    # save best model
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model Saved: {best_model_name}")
    print(f"model path:{model_path}")


if __name__ == "__main__":
    main()
    
    
    