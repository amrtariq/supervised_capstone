import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from pycaret import regression,classification, clustering



# ADD PCA if Availaible
# Add PyCaret

best_used=0
df = pd.DataFrame()
dfOrigin = df

def choose_model():
    global dfOrigin
    #selectType = 
    typeSelect = st.selectbox("Select Mode:",['Other','Clustering'] )
    st.write('For Regresssion/Classification Select Other Clustering is Extra Only')
    if typeSelect == 'Clustering':
        model_select = clustering_ml.ml_select(df)
        return 'Clustering',model_select,'Clustering'
    else:
        targetSelect = st.selectbox("Select Target:", dfOrigin.columns)
        
        if dfOrigin[targetSelect].dtype != "object":
            st.write("Coloumn Selected Should be Dealt with as Regression...")
            #subType = st.selectbox("Select Mode:",['Regression','Classification'] )
            #if(subType == 'Regression'):
            model_select = regression_ml.ml_select(df)
            return targetSelect, model_select,'Regression'
        elif dfOrigin[targetSelect].dtype == "object":
            st.write("Coloumn Selected Should be Dealt with as Classification...")
            model_select = classification_ml.ml_select(df)
            return targetSelect, model_select,'Classification'



        
class regression_ml:
    
    def ml_plot(self,results, best_model):
        # Get the comparison output table as a dataframe
        
        st.write(results)
        regression_plots = ["residuals", "error", "cooks", "rfe", "learning", "vc", "manifold", "feature", "feature_all", "parameter"] 
        plotSelect = st.selectbox("Select Plot:",regression_plots )
    
        plot = regression.plot_model(best_model, plot = plotSelect,display_format='streamlit')
        st.write(plot)
        
    def ml_select(df):
            st.write(df.head())
            
            
            regression_models = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 'mlp', 'xgboost', 'lightgbm', 'catboost']
            model_select = st.multiselect("Select Model or empty for all", regression_models)
            
            
            return  model_select
    @st.cache_data   
    @st.cache_resource
    def ml_train(_self,df, targetSelect, model_select):        

        # Initialize setup
        s = regression.setup(df, session_id= 123, target = targetSelect,iterative_imputation_iters = 10, train_size = 0.8,fold_shuffle= True)
        
        # Compare models
        if (model_select):
            best_model = regression.compare_models(include=model_select)
        else:
            best_model = regression.compare_models()
        global best_used
        best_used = best_model
        
        # Display best model
        st.write("Best model...")
        st.write(best_model)
        results = regression.pull()
        
        plot = regression.plot_model(best_model, plot = 'learning',display_format='streamlit')
        st.write(plot)
        
        return results, best_model

    def ml_init(self,df,targetSelect, model_select):
        
        if(targetSelect):
            st.markdown('Target is '+targetSelect+". Loading ...")
            return self.ml_train(df,targetSelect, model_select)
        
class classification_ml:
    
    def ml_plot(self,results, best_model):
        try:
            # Get the comparison output table as a dataframe
            
            st.write(results)
            classification_plots = ['auc', 'threshold', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary']
            plotSelect = st.selectbox("Select Plot:",classification_plots )
        
            plot = classification.plot_model(best_model, plot = plotSelect,display_format='streamlit')
            st.write(plot)
        except:
            st.write("Plot unaviable please try another one...")
    def ml_select(df):
            st.write(df.head())
            #targetSelect = st.selectbox("Select Target:", df.columns)
            
            classification_models = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost']
            #classification_models = classification.models()
            model_select = st.multiselect("Select Model or empty for all", classification_models)
            
            
            return  model_select
    @st.cache_data   
    @st.cache_resource
    def ml_train(_self,df, targetSelect, model_select):        
        try:
            # Initialize setup
            s = classification.setup(df, fix_imbalance_method='ADASYN', session_id= 123, target = targetSelect, train_size = 0.8,fold_shuffle= False)
            
            # Compare models
            if (model_select):
                best_model = classification.compare_models(include=model_select)
            else:
                best_model = classification.compare_models()
            global best_used
            #best_used = best_model
            st.write("Best model...")
            st.write(best_model)
            results = classification.pull()
            
            return results, best_model
            #plot = regression.plot_model(best_model,display_format='streamlit')
            #st.write(plot)
        except:
            st.write("Model unavailable Please Check Target Data...")
            return 0,0
        
        

    def ml_init(self,df,targetSelect, model_select):
        
        if(targetSelect):
            st.markdown('Target is '+targetSelect+". Loading ...")
            return self.ml_train(df,targetSelect, model_select)
        
class clustering_ml:
    
    def ml_plot(self, best_model):
        # Get the comparison output table as a dataframe
        
        #st.write(results)
        clustering_plots = ['cluster', 'tsne', 'elbow', 'silhouette', 'distance', 'distribution']

        plotSelect = st.selectbox("Select Plot:",clustering_plots )

        plot = clustering.plot_model(best_model, plot = plotSelect,display_format='streamlit')
        st.write(plot)
        st.write("Plot Unavailable Plase Try Another one ...")
        
    def ml_select(df):
        st.write(df.head())
        #targetSelect = st.selectbox("Select Target:", df.columns)
        targetSelect = 0
        clustering_models = ['kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics', 'birch', 'kmodes']
        model_select = st.selectbox("Select Model", clustering_models)
        
        
        return model_select
    @st.cache_data   
    @st.cache_resource
    def ml_train(_self,df, model_select):        

        # Initialize setup
        s = clustering.setup(df, session_id= 123)
        
        # Compare models
        if (model_select):
            best_model = clustering.create_model(model_select)

        #best_used = best_model
        
        # Display best model
        st.write("Best model...")
        st.write(best_model)
        #results = clustering.pull()
        
    
        return best_model

    def ml_init(self,df, model_select):
    
        #st.markdown('Target is '+targetSelect+". Loading ...")
        return self.ml_train(df, model_select)


def draw_pairs(df):
    print("sns not compatible with Pycaret version I have")
    # selections = st.multiselect("Select Columns to Plot:", df.columns)

    # if(selections):
    #     plt.figure()
    #     hue_select = st.selectbox("Select Hue:",df.columns )
    #     if(st.checkbox("Use Hue")):

    #         sns.pairplot(df, vars=selections, hue = hue_select)
    #     else:
    #         sns.pairplot(df, vars=selections)
    #     st.pyplot(plt)


def handle_initial_clean(df):
    global dfOrigin
    st.markdown("### Manual Initial Clean")
    st.write('Select Columns To Delete...')
    #columnToDelete = st.selectbox("Select Coloumn to delete:",options=df.columns)
    cols = st.columns(5)
    count = 0
    for column in df.columns:
        if(cols[count % 5].checkbox(f"{column}")):
            df = df.drop([column], axis=1)
            dfOrigin = dfOrigin.drop([column], axis=1)
        count = count+1
    st.write("\n\n")
    return df


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def convert_categorical(df):
    st.markdown("### Handling Categorical Columns")
    categorical_columns = dfOrigin.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        st.write(f"Handling column: {column}")
        action = st.selectbox(f"Select action for column '{column}':", [
                              "Skip", "Convert to One-Hot", "Use Label-Encoder"])

        if action == "Skip":
            pass
        elif action == "Convert to One-Hot":
            df = pd.get_dummies(df, columns=[column], drop_first=True)
        elif action == "Use Label-Encoder":
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

    st.write("Categorical columns handled according to your selections.")
    return df

def scale_data(df):
    st.markdown("### Handling Continous Columns")
    continous_columns=[]
    for column in dfOrigin.columns:
        if dfOrigin[column].dtype != "object":
            continous_columns.append(column)
    selectedColumns = st.multiselect("Choose Columns to Scale(min-max):",continous_columns)
    sc =MinMaxScaler()
    for c in selectedColumns:
        df[c] = sc.fit_transform(df[[c]])
    

    st.write("Continous columns handled according to your selections.")
    return df


def show_summary_statistics(df):
    st.markdown("### Summary Statistics")

    selected_column = st.selectbox(
        "Select a column for summary statistics", df.columns)
    # Get the data type of the selected column
    data_type = df[selected_column].dtype

    st.write(f"Data Type: {data_type}")
    # Count of unique values
    st.write("Count of Unique Values:", df[selected_column].nunique())

    if data_type == "object":
        st.write("Object Type: Categorical")
    else:
        st.write("Object Type: Numerical")

    st.write(df[selected_column].describe(include = 'object'))

# Load data

def load_data(uploaded_file):

    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    return df


def handle_na_values(df):
    st.markdown("### Handling NA Values")

    columns_with_na = df.columns[df.isnull().any()]
    st.write('to delete column, leave NA then delete in Initial Clean Step...')

    for column in columns_with_na:
        st.write(f"Handling NA values for column: {column}")
        if dfOrigin[column].dtype == "object":
            # Handling categorical columns
            action = st.selectbox(f"Select action for column '{column}':", [
                                  "Drop NA Rows", "Fill with Mode", "Leave as NA"])

            if action == "Drop NA Rows":
                df = df.dropna(subset=[column])
            elif action == "Fill with Mode":
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)
            elif action == "Leave as NA":
                pass
        else:
            # Handling numerical columns
            action = st.selectbox(f"Select action for column '{column}':", [
                                  "Drop NA Rows", "Fill with Mean", "Fill with Median", "Leave as NA"])

            if action == "Drop NA Rows":
                df = df.dropna(subset=[column])
            elif action == "Fill with Mean":
                df[column].fillna(df[column].mean(), inplace=True)
            elif action == "Fill with Median":
                df[column].fillna(df[column].median(), inplace=True)
            elif action == "Leave as NA":
                pass

    st.write("NA values handled according to your selections.")
    return df


def plot_data_hist(df):

    fig, ax = plt.subplots()
    # Example: Create a histogram of a numerical column
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    selected_column = st.selectbox(
        "Select a numeric column for visualization", numeric_columns)
    #fig, ax = plt.subplots()
    ax.hist(df[selected_column], bins=20, edgecolor='k')
    plt.xlabel(selected_column)
    plt.ylabel("Frequency")
    st.pyplot(fig)

    # Option to plot chart with y-axis
    if st.checkbox("Plot Chart with Y-axis"):
        x_column = st.selectbox("Select X-axis column",
                                df.columns, key="x_column_no_sum")
        y_column = st.selectbox("Select Y-axis column",
                                numeric_columns, key="y_column_no_sum")

        fig, ax = plt.subplots()
        ax.plot(df[x_column], df[y_column], alpha=0.5)
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)  # Use the selected Y-axis column as the label
        ax.set_title(f"{x_column} vs {y_column}")
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        # Option to plot chart with y-axis

    # Option to plot bar chart with sum as y-axis
    if st.checkbox("Plot Bar Chart with Sum as Y-axis"):
        x_column = st.selectbox("Select X-axis column",
                                df.columns, key="x_column")
        y_column = st.selectbox(
            "Select Y-axis column for sum", numeric_columns, key="y_column")

        summed_y = df.groupby(x_column)[y_column].sum()
        fig, ax = plt.subplots()
        ax.bar(summed_y.index, summed_y.values, alpha=0.5)
        plt.xlabel(x_column)
        plt.ylabel(f"Sum of {y_column}")
        plt.title(f"{x_column} vs Sum of {y_column}")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        st.pyplot(fig)

    # Option to plot bar chart with top 10 highest sum values
    if st.checkbox("Plot Bar Chart with Top 10 Highest Sum Values"):
        x_column = st.selectbox("Select X-axis column",
                                df.columns, key="x_column_top10")
        y_column = st.selectbox(
            "Select Y-axis column for sum", numeric_columns, key="y_column_top10")

        summed_y = df.groupby(x_column)[y_column].sum()
        # Select top 10 highest sum values
        top_10_summed_y = summed_y.nlargest(10)
        fig, ax = plt.subplots()
        ax.bar(top_10_summed_y.index, top_10_summed_y.values, alpha=0.5)
        plt.xlabel(x_column)
        plt.ylabel(f"Sum of {y_column}")
        plt.title(f"Top 10 {x_column} vs Sum of {y_column}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Option to plot bar chart with top 10 highest sum values
    if st.checkbox("Plot Bar Chart with Top 10 Lowest Sum Values"):
        x_column = st.selectbox("Select X-axis column",
                                df.columns, key="x_column_low10")
        y_column = st.selectbox(
            "Select Y-axis column for sum", numeric_columns, key="y_column_low10")

        summed_y = df.groupby(x_column)[y_column].sum()
        # Select top 10 highest sum values
        low_10_summed_y = summed_y.nsmallest(10)
        fig, ax = plt.subplots()
        ax.bar(top_10_summed_y.index, low_10_summed_y.values, alpha=0.5)
        plt.xlabel(x_column)
        plt.ylabel(f"Sum of {y_column}")
        plt.title(f"Top 10 {x_column} vs Sum of {y_column}")
        plt.xticks(rotation=45)
        st.pyplot(fig)


def main():
    '''
    Main Program Run

    Returns
    -------
    None.

    '''
    rm = regression_ml()
    cl = classification_ml()
    clu = clustering_ml()
    st.markdown("# Exploratory Data Analysis & Model Selection App")
    uploaded_file = st.file_uploader(
        "Upload a CSV or XLSX file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        global df
        global dfOrigin
        df = load_data(uploaded_file)
        dfOrigin = df

        st.markdown("## Preview of the Data")
        st.write(df.head())

        st.markdown("### Initial Data Summary")
        if st.checkbox("Show Summary Statistics"):
            # st.write(df.describe())
            show_summary_statistics(df)

        if st.checkbox("Show & Handle NA/Null Values"):
            na_stats = df.isna().sum()
            st.write("NA Values Statistics & Handling:")
            st.write(na_stats)

            df = handle_na_values(df)

        st.markdown("## Celanup:")
        if st.checkbox("Initial Cleanup"):
            df = handle_initial_clean(df)

        if st.checkbox("Handle Categorical Columns"):
            df_encoded = convert_categorical(df)
            st.markdown("### Encoded Data")
            st.write(df_encoded.head())
            df = df_encoded
        if st.checkbox("Handle Continous Columns"):
            df_encoded = scale_data(df)
            st.markdown("### Encoded Data")
            st.write(df_encoded.head())
            df = df_encoded            
            

        st.markdown("## Data Visualization")
        if st.checkbox("Show Data Visualization"):
            plot_data_hist(df)
        #if st.checkbox("Show Pair Plot"):
            #draw_pairs(df)
        
        if st.checkbox("Model Training"):
            targetSelected, model_selected, typeSelect = choose_model()
            button = st.checkbox('Start Training')
            if(button):
                if typeSelect == 'Regression':
                    results, best_model= rm.ml_init(df,targetSelected, model_selected)
                    rm.ml_plot(results, best_model)
                if typeSelect == 'Classification':
                    results, best_model= cl.ml_init(df,targetSelected, model_selected)
                    cl.ml_plot(results, best_model)
                if typeSelect == 'Clustering':
                    best_model= clu.ml_init(df, model_selected)
                    clu.ml_plot(best_model)
                    st.write("Unable to load plot, please try another one...")

        st.markdown("## Write to File:")
        if st.checkbox("Download Modified Data"):
            # modified_df = df  # Replace this with your actual data modification process

            # Allow user to select file location
            # save_as_csv(modified_df)
            export_file_name = st.text_input(
                "Enter the name for the exported CSV file (without extension):")
            if export_file_name:
                # Save the modified DataFrame to the selected CSV file
                st.download_button(
                    "Press to Download",
                    convert_df(df),
                    export_file_name+".csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.warning("Please enter a valid file name.")
                st.markdown("## Data Visualization")


if __name__ == "__main__":
    main()
