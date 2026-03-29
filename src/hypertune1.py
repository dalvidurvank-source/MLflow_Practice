from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd 
import mlflow

a=load_breast_cancer()
df=pd.DataFrame(data=a.data,columns=a.feature_names)
df["target"]=a.target

X=df.iloc[:,:-1]
y=df["target"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
rf=RandomForestClassifier(random_state=42)

param_grid={"n_estimators":[10,50,100],"max_depth":[None,10,20,30]}

grid_search=GridSearchCV(estimator=rf,cv=5,param_grid=param_grid,n_jobs=-1,verbose=2)

# grid_search.fit(X_train,y_train)

# best_param=grid_search.best_params_
# best_score=grid_search.best_score_

# print(best_param)
# print(best_score)
mlflow.set_experiment("New_Experiment")
with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric("accuracy", best_score)

    # Log training data
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    # Log test data
    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    # Log source code
    mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

    # Set tags
    mlflow.set_tag("author", "Vikash Das")

    print(best_params)
    print(best_score)
  


