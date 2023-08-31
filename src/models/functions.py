def process_data(df_data):
    pattern = '^(\d{1,2}-[A-Z][a-z]{2})|([A-Z][a-z]{2}-\d{1,2})$'
    replacement = df['ht'].mode()[0]
    def replace_non_matching(item):
        return item if re.match(pattern, str(item)) else replacement
    df_data['ht'] = df_data['ht'].apply(replace_non_matching)

    df_data.drop(['Rec_Rank', 'dunks_ratio', 'pick', 'type', 'num', 'player_id'], inplace=True, axis=1)
    valid_yr_values = ['So', 'Sr', 'Jr', 'Fr']
    df_data['yr'].replace(list(set(df_data['yr'].unique()) - set(valid_yr_values)),df_data['yr'].mode()[0], inplace=True)
    
    null_cols = list(df_data.columns[df_data.isnull().any()])
    for col in null_cols:
        if col in num_cols:
            df_data[col].fillna(df_data[col].mean(), inplace=True)
        else:
            df_data[col].fillna(df_data[col].mode(), inplace=True)
    return df_data

def fit_predict_proba(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_train_pred_prob = model.predict_proba(X_train)[:,1]
    y_val_pred_prob = model.predict_proba(X_val)[:,1]
    print('The AUROC value for the training set is: ', roc_auc_score(y_train, y_train_pred_prob))
    print('The AUROC value for the validation set is: ', roc_auc_score(y_val, y_val_pred_prob))