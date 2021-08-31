df = pd.DataFrame.from_dict(label_gen(),orient='index').reset_index()
df.columns = ['recording_id']+['species_'+str(n) for n in range(24)]
label_cols = [f"species_"+str(m) for m in range(24)]

mst = MultilabelStratifiedKFold(n_splits=5)
X = df.recording_id.values
y = df[label_cols].values
df['kfold'] = -1
for i,(train_ind,val_ind) in enumerate(mst.split(X,y)):

    df.loc[val_ind,'kfold'] = i

df.to_csv("RFCX_kfold.csv",index=False)



dur_sample = []
data = pd.read_csv("RFCX_kfold.csv")
org_data = pd.read_csv("/home/lustbeast/AudioClass/Dataset/rfcx-species-audio-detection/train_tp.csv")
org_data = org_data.groupby("recording_id").agg({'t_min':lambda x:min(x),'t_max':lambda x:max(x)}).reset_index()
org_data['duration'] = org_data['t_max'] - org_data["t_min"]
org_data['duration'] = np.ceil(org_data.duration.values)+5
data = data.merge(org_data,on="recording_id",how='left')
data['dur_sample'] = -1
for i in tqdm(range(len(data))):
    song_name = os.path.join("/home/lustbeast/AudioClass/Dataset/rfcx-species-audio-detection/train",data.loc[i,"recording_id"]+".flac")
    song,sr = librosa.load(song_name,sr=None)
    durs = song.shape[0] / sr
    data['dur_sample'].iloc[i] = durs
data.to_csv("RFCX_kfold.csv")
