from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from IPython.display import clear_output

def tsne_astrometric(features, df):
    df_astrometric = df[features].copy()
    
    # Step 2: Drop rows with any NaN in those features
    df_astrometric_clean = df_astrometric.dropna()
    df_subset = df.loc[df_astrometric_clean.index]  # retain full info for color, label_group etc.
    
    # Step 3: Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_astrometric_clean)
    
    # Step 4: Run t-SNE on the cleaned & scaled data
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)
    
    # Step 5: Add t-SNE results to df
    df_subset["tsne_astrometric_1"] = tsne_result[:, 0]
    df_subset["tsne_astrometric_2"] = tsne_result[:, 1]

    return df_subset
    
def get_silhouette_scores(featurelist,
                          df,
                          op=lambda dfthis, lblfld, thresh: (dfthis[lblfld] > thresh).astype(int),
                          labelfield="score_weighted_root_sumnorm",
                          binary_threshold=0.6):
    all_comb_features = []
    for r in range(2, len(featurelist)+1):
        all_comb_features.extend(combinations(featurelist, r))

    max_score = 0
    max_score_feature = None
    ret_dict = {}
    for this_feature in tqdm(all_comb_features, desc="Processing Silhouette Score for Feature Set"):
        print(f"Number of feature sets: {len(all_comb_features)}\n")
        print(f"Max Score: {max_score} \twith feature set: {max_score_feature}")
        this_feature = list(this_feature)
        df_tsne = tsne_astrometric(this_feature, df)
        labels = op(df_tsne, labelfield, binary_threshold)
        silh_score = silhouette_score(df_tsne[["tsne_astrometric_1", "tsne_astrometric_2"]], labels)
        ret_dict[str(this_feature)] = silh_score
        print(silh_score)
        if silh_score > max_score:
            max_score = silh_score
            max_score_feature = this_feature
        clear_output(wait=True)
    return ret_dict