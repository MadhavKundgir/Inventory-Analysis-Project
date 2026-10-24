from data_preprocessing import scaler_features,apply_lables,load_invoice_data,split_data
from modeling_evaluation import train_random_forest,evaluate_classifier 
import joblib

features = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

target = 'flag_invoice'

def main():
    
    # load_data
    df = load_invoice_data()
    df = apply_lables(df)

    # preper data 
    X_train, X_test, y_train, y_test = split_data(df,features,target)
    X_train_scaled, X_test_scaled = scaler_features(
        X_train, X_test, 'models/scaler.pkl'
    )

    # Train and evaluate models 
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
     grid_search.best_estimator_,
     X_test_scaled,
     y_test,
     "Random Forest Classifier"
    )

    # save the best model

    joblib.dump(grid_search.best_estimator_,'models/predict_flag_invoice.pkl')

if __name__ == "__main__":
    main()
