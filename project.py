import streamlit as st
import pandas as pd
st.title("Report")
m = [['NuSVC', 0.90, 0.90, 0.90, 0.90, 4.90],
['SVC', 0.90, 0.90, 0.90, 0.90, 3.17],
['RidgeClassifierCV', 0.89, 0.89, 0.89, 0.89, 0.79],
['LinearDiscriminantAnalysis', 0.88, 0.88, 0.88, 0.88, 0.97],
['RidgeClassifier', 0.88, 0.88, 0.88, 0.88, 0.17],
['LGBMClassifier', 0.88, 0.88, 0.88, 0.88, 18.79],
['XGBClassifier', 0.88, 0.88, 0.88, 0.88, 17.20],
['LogisticRegression', 0.86, 0.86, 0.86, 0.86, 0.36],
['ExtraTreesClassifier', 0.85, 0.86, 0.86, 0.85, 1.42],
['QuadraticDiscriminantAnalysis', 0.85, 0.85, 0.85, 0.85, 0.94],
['CalibratedClassifierCV', 0.85, 0.85, 0.85, 0.85, 7.84],
['PassiveAggressiveClassifier', 0.85, 0.85, 0.85, 0.85, 0.29],
['AdaBoostClassifier', 0.84, 0.84, 0.84, 0.84, 17.36],
['Perceptron', 0.84, 0.84, 0.84, 0.84, 0.23],
['KNeighborsClassifier', 0.84, 0.84, 0.84, 0.84, 0.34],
['RandomForestClassifier', 0.84, 0.84, 0.84, 0.84, 8.08],
['SGDClassifier', 0.82, 0.82, 0.82, 0.82, 0.54],
['LinearSVC', 0.82, 0.82, 0.82, 0.82, 2.31],
['BaggingClassifier', 0.80, 0.80, 0.80, 0.80, 26.71],
['GaussianNB', 0.76, 0.76, 0.76, 0.76, 0.13],
['NearestCentroid', 0.76, 0.76, 0.76, 0.76, 0.15],
['BernoulliNB', 0.75, 0.75, 0.75, 0.75, 0.19],
['DecisionTreeClassifier', 0.73, 0.73, 0.73, 0.73, 4.05],
['ExtraTreeClassifier', 0.68, 0.68, 0.68, 0.68, 0.12],
['LabelSpreading', 0.50, 0.50, 0.50, 0.34, 1.05],
['LabelPropagation', 0.50, 0.50, 0.50, 0.34, 0.78],
['DummyClassifier', 0.50, 0.50, 0.50, 0.33, 0.11]]







# Choice
menu = ["Object Classification", "Object Detection", "Text Classification"]
choice = st.sidebar.selectbox('Content', menu)

if choice == 'Object Classification':   
    st.subheader("Image Classification")
    st.write('### Fish 🐟 | Glass 🥂 | Metal_Battery 🔋 | Plastic 🏭')
    col1, col2 = st.columns(2)

    with col1:
        st.image('image/cephalopholis_cyanostigma_8.jpg', caption = 'Fish 🐟')

    with col2:
        st.image('image/brown-glass7.jpg', caption = 'Glass 🥂')

    col3, col4 = st.columns(2)
    with col3:
        st.image('image/battery201.jpg', caption = 'Metal_Battery 🔋')

    with col4:
        st.image('image/new_53I3SC0WUB17.jpg', caption = 'Plastic 🏭')
    st.write("### 1. CNN - 2 Classes: FISH - GARBAGE")
    st.write('Dataset: ~', 6000, 'images')
    st.write('Total params: ', 5409665)
    st.write('Training time: ', 45, 'minutes')
    tab_acc, tab_confusion_matrix, tab_classification_report = st.tabs(["Accuracy & Loss", "Confusion Matrix", "Classification Report"])
    with tab_acc:
        st.markdown('***Accuracy & Loss***')
        st.write('Train accuracy: ', 0.9882, 'Train loss: ', 0.0319)
        st.write('Test accuracy: ', 0.9708, 'Test loss: ', 0.1034)
    with tab_confusion_matrix:
        st.markdown('***Confusion Matrix***')
        st.image('image/new_2_Classes_CNN_CF.jpg', width = 500)
    with tab_classification_report:
        st.markdown('***Classification Report***')
        st.image('image/new_2_Classes_CNN_R.jpg', width = 600)

    st.write("### 2. CNN - 5 Classes: FISH - GLASS - METAL_BATTERY - PLASTIC - OTHERS")
    st.write('Total params: ', 3735717)
    st.write('Training time: ', 3, 'hours')
    st.write('Dataset: ~', 6000, 'images')
    tab_acc, tab_confusion_matrix, tab_classification_report = st.tabs(["Accuracy & Loss", "Confusion Matrix", "Classification Report"])
    with tab_acc:
        st.markdown('***Accuracy & Loss***')
        st.write('Train accuracy: ', 0.9519, 'Train loss: ', 0.1284)
        st.write('Test accuracy: ', 0.8637, 'Test loss: ', 0.4434)
    with tab_confusion_matrix:
        st.markdown('***Confusion Matrix***')
        st.image('image/new_5_Classes_CNN_CF.jpg', width = 500)
    with tab_classification_report:
        st.markdown('***Classification Report***')
        st.image('image/new_5_Classes_CNN_R.jpg', width = 600)

    st.write("### 3. CNN & INCEPTION_V3 - 5 Classes: FISH - GLASS - METAL_BATTERY - PLASTIC - OTHERS")
    st.write('Dataset: ~', 6000, 'images')
    st.write('Total params: ', 298576260)
    st.write('Training time: ', 1, 'hours', 52, 'minutes')
    tab_acc, tab_confusion_matrix, tab_classification_report = st.tabs(["Accuracy & Loss", "Confusion Matrix", "Classification Reprt"])
    with tab_acc:
        st.markdown('***Accuracy & Loss***')
        st.write('Train accuracy: ', 0.9519, 'Train loss: ', 0.1284)
        st.write('Test accuracy: ', 0.8637, 'Test loss: ', 0.4434)
    with tab_confusion_matrix:
        st.markdown('***Confusion Matrix***')
        st.image('image/new_5_Classes_pretrained_CF.jpg', width = 500)
    with tab_classification_report:
        st.markdown('***Classification Report***')
        st.image('image/new_5_Classes_pretrained_R.jpg', width = 600)


elif choice == 'Object Detection':
    st.subheader("Fish | Garbage")

    st.write("### 1. YOLO V4")
    st.write('Dataset: ~', 5000, 'images')
    st.write('Training time: ', 6, 'hours', 11, 'minutes')
    tab_chart, tab_score= st.tabs(["Chart", "Score"])
    with tab_score:
        st.markdown('***Cunfusion matrix & Classification Report***')
        st.image('image/yolo_v4_score.jpg', width = 700)
    with tab_chart:
        st.markdown('***Training Process***')
        col1, col2 = st.columns(2)
        with col1:
            st.image('image/chart_yolov4-custom.png')
        with col2:
            st.image('image/v4_chart.png')

    st.write("### 2. YOLO V7")
    st.write('Dataset: ~', 5000, 'images')
    st.write('Training time: ', 11, 'hours')
    tab_detail, tab_score= st.tabs(["Detail score", "Score"])
    with tab_score:
        st.markdown('***Cunfusion matrix & Classification Report***')
        st.image('image/v7.jpg', width = 700)
    with tab_detail:
        st.markdown('***Training Process***')
        col1, col2 = st.columns(2)
        with col1:
            st.image('image/F1_curve.png', caption = 'F1_curve')
        with col2:
            st.image('image/P_curve.png', caption = 'P_curve')

        col3, col4 = st.columns(2)
        with col3:
            st.image('image/PR_curve.png', caption = 'PR_curve')
        with col4:
            st.image('image/R_curve.png', caption = 'R_curve')
        
        col5, col6 = st.columns(2)
        with col5:
            st.image('image/results.png', caption = 'results')
        with col6:
            st.image('image/confusion_matrix.png', caption = 'confusion_matrix')

elif choice == 'Text Classification':
    st.write("## 1) 2 Classes: POSITIVE - NEGATIVE")
    st.write("### *1.1 RNN/LSTM*")
    st.write('Dataset: ~', 84000, 'records')
    tab_acc, tab_report = st.tabs(["Accuracy & Loss", "Detail"])
    
    with tab_acc:
        st.markdown('***Accuracy & Loss***')
        st.write('Train accuracy: ', 0.9681, 'Train loss: ', 0.1059)
        st.write('Test accuracy: ', 0.9402, 'Test loss: ', 0.15)
    with tab_report:
        st.markdown('***Confusion Matrix***')
        df12 = pd.DataFrame([[1387, 251], [95, 4184]])
        st.dataframe(df12)

        st.markdown('***Classification Report***')
        st.image('image/new_2_classes_LSTM.jpg', width = 600)


    st.write("### *1.2 PhoBERT*")
    st.write('Dataset: ~', 84000, 'records')
    tab_acc, tab_report = st.tabs(["Accuracy & Loss", "Detail"])
    
    with tab_acc:
        st.markdown('***Accuracy & F1 Score***')
        st.write('Train accuracy: ', 0.9902, 'Average F1 Score: ', 0.9900)
        st.write('Test accuracy: ', 0.9340, 'Average F1 Score: ', 0.93)
    with tab_report:
        st.markdown('***Confusion Matrix***')
        df11 = pd.DataFrame([[1238, 83], [118, 1608]])
        st.dataframe(df11)

        st.markdown('***Classification Report***')
        st.image('image/new_2_classes_phoBERT.jpg', width = 600)

    st.write("### *1.3 PhoBERT for Feature Extraction*")
    st.write('Dataset: ~', 2500, 'records')
    tab_acc, tab_report, tab_lazy = st.tabs(["SVC Accuracy", "SVC Detail", 'Lazy Predict'])
    with tab_lazy:
        df2 = pd.DataFrame(m, columns = ['Model', 'Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1 Score', 'Time Taken']).set_index('Model')
        st.dataframe(df2)

    with tab_acc:
        st.markdown('***Accuracy***')
        st.write('Train accuracy: ', 0.94725)
        st.write('Test accuracy: ', 0.9)
    with tab_report:
        st.markdown('***Confusion Matrix***')
        df1 = pd.DataFrame([[450, 54], [46, 450]])
        st.dataframe(df1)
        st.markdown('***Classification Report***')
        st.image('image/new_2_classes_extract_phoBERT.jpg', width = 600)














