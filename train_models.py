"""Generate synthetic raw-material datasets and train ten classifiers."""
import json, os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)
ROOT=os.path.dirname(os.path.abspath(__file__))
MODELS_DIR=os.path.join(ROOT,"models")

def make_data(industry,n=800):
    rows=[]
    for _ in range(n):
        if industry=="Food Processing":
            x=[np.random.uniform(5,25),np.random.uniform(8,35),np.random.uniform(1,20),np.random.uniform(.5,15),np.random.uniform(.5,8),np.random.uniform(3.5,8.5),np.random.uniform(1,30),np.random.uniform(.1,3),np.random.uniform(1,10),np.random.uniform(2,35)]
            score=-abs(x[0]-12)*.5+x[1]*.4-x[2]*.2+x[3]*.3-abs(x[5]-6.5)*.8-x[6]*.1-x[7]*1.2+x[8]*.2-abs(x[9]-15)*.3+np.random.normal(0,1.5)
            y="Premium" if score>8 else "Good" if score>3 else "Fair" if score>-2 else "Poor"
            cols=["moisture_%","protein_%","fat_%","fiber_%","ash_%","ph_level","sugar_%","acidity_%","color_index","storage_temp_c"]
        elif industry=="Textile":
            x=[np.random.uniform(15,40),np.random.uniform(20,50),np.random.uniform(2.5,6.5),np.random.uniform(75,95),np.random.uniform(3,12),np.random.uniform(5,14),np.random.uniform(0,8),np.random.uniform(1,10),np.random.uniform(.6,1),np.random.uniform(5,25)]
            score=x[0]*.5+x[1]*.4-abs(x[2]-4)*2+x[3]*.3+x[4]*.2-x[6]*1.5+x[7]*.3+x[8]*5-x[9]*.4+np.random.normal(0,2)
            y="A-Grade" if score>30 else "B-Grade" if score>20 else "C-Grade" if score>10 else "Rejected"
            cols=["fiber_length_mm","fiber_strength_gtex","micronaire","uniformity_%","elongation_%","moisture_regain_%","trash_content_%","color_grade","maturity_ratio","short_fiber_%"]
        else:
            x=[np.random.uniform(85,99.9),np.random.uniform(100,5000),np.random.uniform(4,8),np.random.uniform(.1,50),np.random.uniform(.1,15),np.random.uniform(0,20),np.random.uniform(150,260),np.random.uniform(1,10),np.random.uniform(0,5),np.random.uniform(0,1000)]
            score=x[0]*.6-abs(x[2]-5.5)*3-x[3]*.2-x[4]*.5-x[5]*1.5+x[7]*.5-x[8]*2-np.log1p(x[9])*2+np.random.normal(0,2)
            y="Certified" if score>50 else "Approved" if score>40 else "Conditional" if score>30 else "Rejected"
            cols=["purity_%","viscosity_mPas","ph_level","particle_size_um","moisture_%","peroxide_value","saponification_value","color_degree","heavy_metals_ppm","microbial_cfu"]
        rows.append(x+[y])
    return pd.DataFrame(rows,columns=cols+["quality_grade"])

MODELS={
"Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
"Decision Tree":DecisionTreeClassifier(max_depth=10,random_state=42),
"Random Forest":RandomForestClassifier(n_estimators=100,random_state=42),
"Gradient Boosting":GradientBoostingClassifier(n_estimators=100,random_state=42),
"AdaBoost":AdaBoostClassifier(n_estimators=100,random_state=42),
"Extra Trees":ExtraTreesClassifier(n_estimators=100,random_state=42),
"Bagging":BaggingClassifier(n_estimators=50,random_state=42),
"Support Vector Machine":SVC(kernel="rbf",probability=True,random_state=42),
"K-Nearest Neighbors":KNeighborsClassifier(n_neighbors=7),
"Naive Bayes":GaussianNB()}

def train(industry,df):
    X=df.drop(columns="quality_grade"); y=df["quality_grade"]
    le=LabelEncoder(); ye=le.fit_transform(y); scaler=StandardScaler(); xs=scaler.fit_transform(X)
    xtr,xte,ytr,yte=train_test_split(xs,ye,test_size=.2,random_state=42)
    trained={}; results={}
    for name,model in MODELS.items():
        model.fit(xtr,ytr); acc=accuracy_score(yte,model.predict(xte)); cv=cross_val_score(model,xs,ye,cv=5)
        trained[name]=model; results[name]={"accuracy":round(acc*100,2),"cv_mean":round(cv.mean()*100,2),"cv_std":round(cv.std()*100,2)}
        print(industry,name,results[name])
    rf=trained["Random Forest"]
    return scaler,le,trained,{"model_results":results,"features":list(X.columns),"classes":le.classes_.tolist(),"feature_imp":dict(zip(X.columns,[round(v*100,2) for v in rf.feature_importances_])),"class_dist":df.quality_grade.value_counts().to_dict(),"best_model":max(results,key=lambda k:results[k]["accuracy"])}

def main():
    os.makedirs(MODELS_DIR,exist_ok=True); summary={}
    for industry in ["Food Processing","Textile","Cosmetics"]:
        key=industry.lower().replace(" ","_"); df=make_data(industry); df.to_csv(os.path.join(MODELS_DIR,key+"_dataset.csv"),index=False)
        scaler,le,models,info=train(industry,df)
        for suffix,obj in [("scaler",scaler),("le",le),("models",models)]:
            with open(os.path.join(MODELS_DIR,f"{key}_{suffix}.pkl"),"wb") as f: pickle.dump(obj,f)
        summary[industry]=info
    with open(os.path.join(MODELS_DIR,"summary.json"),"w") as f: json.dump(summary,f,indent=2)
    print("Training complete. Run: python app.py")

if __name__=="__main__": main()
