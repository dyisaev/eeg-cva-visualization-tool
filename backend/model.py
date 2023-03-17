import torch
import torch.nn.functional as F
import pickle
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


def balanced_prevalence_sampling(model,x,frames,num=256,prevalence_threshold=0.5):
    probs=torch.squeeze(F.softmax(model(x.to(device)),dim=1))
    confidences = probs[:,-1]

    sample_weights=torch.full_like(confidences,fill_value=0)
    bin_boundaries = torch.tensor([0,prevalence_threshold,1.01])
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.ge(bin_lower.item()) * confidences.lt(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            sample_weights[in_bin]=prop_in_bin.item() 
    mask=(sample_weights!=0)
    sample_weights[mask]=1.0/(sample_weights[mask]/sample_weights.sum())
    sample_idx=torch.multinomial(sample_weights,num)

    mask_for_idx=torch.ones(sample_weights.shape)
    mask_for_idx[sample_idx]=0
    mask_for_idx=mask_for_idx==1
    return frames.iloc[sample_idx.detach().cpu().numpy(),0].tolist()

class Model:
    def __init__(self,model_filename,preprocess_filename=None) -> None:
        self.model = torch.load(model_filename)
        self.model = self.model.to(device)
        self.preprocess_filename=preprocess_filename
        #specifically my case - preprocess parames have a specific convention
        self.preprocess_filename=model_filename+'_preprocess.pickle'
        self.preprocess_params = pickle.load(open(self.preprocess_filename,'rb'))
        self.active_learning_labeled_frames=[]
        self.active_learning_labels=[]
        self.optim = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.8)
        self.n_training_epochs=20
        self.criterion=torch.nn.CrossEntropyLoss()

    def preprocess(self,df,infer=True): 
        feat_list=['frame', 'gazex', 'gazey','yaw', 'pitch', 'roll','nosex', 'nosey'] 
        df_ml=df[feat_list]
        if infer:
            gazex_median,gazey_median,yaw_median,pitch_median,roll_median,nosex_median,nosey_median,ear_median,OHEncoder = self.preprocess_params       
        else:
            gazex_median=df_ml['gazex'].median()
            gazey_median=df_ml['gazey'].median()
            yaw_median=df_ml['yaw'].median()
            pitch_median=df_ml['pitch'].median()
            roll_median=df_ml['roll'].median()
            nosex_median=df_ml['nosex'].median()
            nosey_median=df_ml['nosey'].median()
        df_ml['gazex_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['gazex']-gazex_median)
        df_ml['gazex_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['gazex']+gazex_median)

        df_ml['gazey_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['gazey']-gazey_median)
        df_ml['gazey_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['gazey']+gazey_median)


        df_ml['yaw_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['yaw']-yaw_median)
        df_ml['yaw_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['yaw']+yaw_median)

        df_ml['pitch_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['pitch']-pitch_median)
        df_ml['pitch_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['pitch']+pitch_median)

        df_ml['roll_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['roll']-roll_median)
        df_ml['roll_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['roll']+roll_median)

        df_ml['nosex_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['nosex']-nosex_median)
        df_ml['nosex_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['nosex']+nosex_median)

        df_ml['nosey_plus']=np.maximum(np.repeat(0,df_ml.shape[0]),df_ml['nosey']-nosey_median)
        df_ml['nosey_minus']=np.maximum(np.repeat(0,df_ml.shape[0]),-df_ml['nosey']+nosey_median)
        df_ml=df_ml.dropna()
        data=df_ml[['gazex_plus', 'gazex_minus','gazey_plus', 'gazey_minus','yaw_plus', 'yaw_minus','pitch_plus', 'pitch_minus','roll_plus', 'roll_minus',
                'nosex_plus', 'nosex_minus','nosey_plus', 'nosey_minus']]
        data = data.to_numpy()

        # in case One-Hot Encoder is present in the preprocessing  (meaning the model was trained with One-Hot Encoding of participants)
        if OHEncoder is not None:
            oh_infer=np.zeros((1,OHEncoder.categories_[0].shape[0]+1))
            oh_infer[0,-1]=1
            id_onehot=np.repeat(oh_infer,data.shape[0],axis=0)
            data=np.concatenate((id_onehot,data),axis=1)
        return df_ml[['frame']],data

    def load_dataset(self,dataset):
        self.dataset=dataset
        return
    def predict(self):
        frames,data=self.preprocess(self.dataset,infer=True)
        x=torch.FloatTensor(data)  
        pred=torch.squeeze( F.softmax(self.model(x.to(device)),dim=1))
        return frames['frame'].to_numpy(),pred.detach().cpu().numpy()[:,1]
    def train(self):

        frames,data=self.preprocess(self.dataset,infer=True)
        frames_filtered = frames[frames['frame'].isin(self.active_learning_labeled_frames)]
        data_filtered = data[frames['frame'].isin(self.active_learning_labeled_frames)]
        y=np.stack([(np.array(self.active_learning_labels)==0).astype(int),(np.array(self.active_learning_labels)!=0).astype(int)],axis=1)
        x=torch.FloatTensor(data_filtered).to(device)
        y=torch.Tensor(y.astype(int)).to(device)
        for epoch in range (self.n_training_epochs):
            outputs = self.model(x)
            loss = self.criterion(torch.squeeze(outputs), y) 
            self.optim.zero_grad() 
            loss.backward() 
            self.optim.step()
            print(f'epoch {epoch} loss {loss}')
        self.scheduler.step()
    def sample(self,data,frames,num):
        PREVALENCE_THRESHOLD=0.14
        x=torch.FloatTensor(data)
        return balanced_prevalence_sampling(self.model,x,frames,num,prevalence_threshold=PREVALENCE_THRESHOLD)
    def generate_al_batch(self,batch_size):
        frames,data=self.preprocess(self.dataset,infer=True)


        frames_filtered = frames[~frames['frame'].isin(self.active_learning_labeled_frames)]

        data_filtered = data[~frames['frame'].isin(self.active_learning_labeled_frames)]

        self.frames_to_label = self.sample (data_filtered,frames_filtered,batch_size)
        return self.frames_to_label
    def label(self, labeled_frames):
        for frame,y in labeled_frames:
            if frame not in self.active_learning_labeled_frames:
                self.active_learning_labeled_frames.append(frame)
                self.active_learning_labels.append(y)
        return
    def save(self,filename):
        pass
