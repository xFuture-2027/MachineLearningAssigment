"""Reproduce Assessment 4 exploratory figures and a prepared, unencoded CSV.
Run: python analyse_data.py. Does not refit models or change test predictions.
Feature association charts use development data only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import load_data, plot_model_comparison
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

ROOT=Path(__file__).resolve().parent

def main():
    X,y,groups,source=load_data()
    manifest=pd.read_csv(ROOT/'results/split_manifest.csv',dtype={'group':str})
    assert np.array_equal(manifest.row_id.to_numpy(),np.arange(len(X)))
    assert np.array_equal(manifest.group.to_numpy(),groups.astype(str))
    dev=manifest.split.eq('train').to_numpy()
    out=ROOT/'results/assessment4';out.mkdir(parents=True,exist_ok=True)
    prepared=X.copy();prepared.insert(0,'row_id',np.arange(len(X)))
    prepared['phishing']=y.to_numpy();prepared['split']=manifest.split.to_numpy()
    prepared.to_csv(out/'prepared_dataset.csv',index=False)
    # row_id and split are metadata, NEVER input features.
    features=['SSLfinal_State','URL_of_Anchor','having_Sub_Domain','SFH']
    rows=[]
    for f in features:
        z=pd.DataFrame({'code':X.loc[dev,f],'phishing':y.loc[dev]})
        s=z.groupby('code').phishing.agg(['count','sum','mean']).reset_index()
        s.insert(0,'feature',f);rows.append(s)
    conditional=pd.concat(rows,ignore_index=True)
    conditional.to_csv(out/'development_category_rates.csv',index=False)
    cross=pd.crosstab(X.loc[dev,'SSLfinal_State'],X.loc[dev,'URL_of_Anchor'])
    rates=pd.crosstab(X.loc[dev,'SSLfinal_State'],X.loc[dev,'URL_of_Anchor'],values=y.loc[dev],aggfunc='mean')
    cross.to_csv(out/'development_joint_counts.csv');rates.to_csv(out/'development_joint_rates.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,3.4))
    vals=[int((1-y).sum()),int(y.sum())]
    axs[0].bar(['Legitimate','Phishing'],vals,color=['#26465C','#C06C45'])
    for i,v in enumerate(vals):axs[0].text(i,v+100,f'{v:,}',ha='center')
    axs[0].set_ylim(0,7000);axs[0].set_ylabel('Records');axs[0].set_title('Recorded class balance')
    vals=[len(X.drop_duplicates()),int(X.duplicated().sum())]
    axs[1].bar(['First occurrence\nof each pattern','Additional\noccurrences'],vals,color=['#26465C','#679B94'])
    for i,v in enumerate(vals):axs[1].text(i,v+100,f'{v:,}',ha='center')
    axs[1].set_ylim(0,7000);axs[1].set_ylabel('Records');axs[1].set_title('Exact input repetition')
    fig.tight_layout();fig.savefig(out/'01_data_audit.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,6))
    for ax,f in zip(axs.ravel(),features):
        s=conditional[conditional.feature.eq(f)]
        ax.bar(s.code.astype(str),s['mean']*100,color='#26465C')
        for i,(_,r) in enumerate(s.iterrows()):ax.text(i,r['mean']*100+3,f"n={int(r['count']):,}",ha='center',fontsize=9)
        ax.set_ylim(0,120);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Phishing labels (%)');ax.set_title(f);ax.set_xlabel('Original category code')
    fig.tight_layout();fig.savefig(out/'02_category_rates.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.3));im=ax.imshow(rates*100,cmap='Blues',vmin=0,vmax=100)
    for i in range(len(rates.index)):
        for j in range(len(rates.columns)):
            v=rates.iloc[i,j];n=cross.iloc[i,j]
            ax.text(j,i,f'{v*100:.1f}%\nn={n:,}' if n else 'No records',ha='center',va='center',color='white' if v>.55 else 'black')
    ax.set_xticks(range(len(rates.columns)),rates.columns);ax.set_yticks(range(len(rates.index)),rates.index)
    ax.set_xlabel('URL_of_Anchor category');ax.set_ylabel('SSLfinal_State category')
    fig.colorbar(im,ax=ax,label='Phishing labels (%)');fig.tight_layout();fig.savefig(out/'03_joint_rates.png',dpi=180);plt.close(fig)
    m=pd.read_csv(ROOT/'results/metrics.csv');c=pd.read_csv(ROOT/'results/cv_summary.csv')
    core=m.iloc[:4];fig,ax=plt.subplots(figsize=(9,3.7));xx=np.arange(4)
    ax.bar(xx-.18,c.iloc[:4].cv_f1*100,width=.36,label='Development CV',color='#679B94')
    ax.bar(xx+.18,core.f1*100,width=.36,label='Held-out test',color='#26465C')
    for i in range(4):
        for off,v in [(-.18,c.iloc[i].cv_f1*100),(.18,core.iloc[i].f1*100)]:ax.text(i+off,v+1,f'{v:.2f}',ha='center',fontsize=9)
    ax.set_xticks(xx,['Decision Tree','Random Forest','Categorical NB','Multilayer ANN']);ax.set_ylim(0,110);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Phishing F1 (%)');ax.legend(loc='lower right');fig.tight_layout();fig.savefig(out/'04_model_f1.png',dpi=180);plt.close(fig)
    # Describe saved predictions only: no retraining or threshold selection.
    all_metrics=pd.concat([m,pd.read_csv(ROOT/'results/deep_learning_metrics.csv')],ignore_index=True)
    ranked=all_metrics.sort_values('f1',ascending=True).reset_index(drop=True)
    aliases={'Deep ANN (3 layers)':'Deep ANN', 'k-Nearest Neighbours':'k-NN',
             'LSTM (bidirectional)':'LSTM', 'Simple RNN (bidirectional)':'RNN',
             'Top 10 Random Forest':'Reduced RF'}
    fig,ax=plt.subplots(figsize=(9,5.7)); yy=np.arange(len(ranked))
    colors=['#679B94' if name=='Random Forest' else '#26465C' for name in ranked.model]
    ax.barh(yy,ranked.f1*100,color=colors)
    for i,r in ranked.iterrows():
        ax.plot([r.f1_ci_low*100,r.f1_ci_high*100],[i,i],color='#C06C45',lw=1.6)
        ax.text(max(r.f1_ci_high*100+1,1),i,f'{r.f1*100:.2f}%',va='center',fontsize=9)
    ax.set_yticks(yy,[aliases.get(x,x) for x in ranked.model]);ax.set_xlim(0,108)
    ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('Held-out phishing F1 (%)')
    ax.set_title('All saved comparisons with 95% group-bootstrap intervals')
    ax.text(3,-.42,'Zero F1; accuracy 55.70%',fontsize=9)
    fig.tight_layout();fig.savefig(out/'05_all_model_f1.png',dpi=180);plt.close(fig)
    rf=m.loc[m.model.eq('Random Forest')].iloc[0]
    counts=np.array([[rf.TN,rf.FP],[rf.FN,rf.TP]],dtype=int)
    percentages=counts/counts.sum(axis=1,keepdims=True)*100
    fig,ax=plt.subplots(figsize=(7.2,4.2));ax.imshow(percentages,cmap='Blues',vmin=0,vmax=100)
    names=np.array([['True negatives','False alarms'],['Missed phishing','True positives']])
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{names[i,j]}\n{counts[i,j]:,} records\n{percentages[i,j]:.2f}% within class',ha='center',va='center',color='white' if percentages[i,j]>55 else '#10213A',fontsize=9)
    ax.set_xticks([0,1],['Legitimate','Phishing']);ax.set_yticks([0,1],['Legitimate','Phishing'])
    ax.set_xlabel('Predicted class');ax.set_ylabel('Actual class');ax.set_title('Random Forest held-out errors')
    fig.tight_layout();fig.savefig(out/'06_forest_confusion.png',dpi=180);plt.close(fig)
    predictions=pd.read_csv(ROOT/'results/predictions.csv',float_precision='round_trip')
    fig,axs=plt.subplots(1,2,figsize=(10,4.2));ap_rows=[]
    for name,color in [('Random Forest','#679B94'),('SVM (RBF)','#26465C'),('Gradient Boosting','#C06C45')]:
        pred=predictions.loc[predictions.model.eq(name)]
        fpr,tpr,_=roc_curve(pred.actual,pred.phishing_score)
        precision,recall,_=precision_recall_curve(pred.actual,pred.phishing_score)
        auc=float(m.loc[m.model.eq(name),'roc_auc'].iloc[0]);ap=average_precision_score(pred.actual,pred.phishing_score)
        axs[0].plot(fpr,tpr,label=f'{name}: AUC {auc:.4f}',color=color,lw=1.6)
        axs[1].plot(recall,precision,label=f'{name}: AP {ap:.4f}',color=color,lw=1.6)
        ap_rows.append({'model':name,'average_precision':ap,'roc_auc':auc})
    prevalence=float(pred.actual.mean())
    axs[0].plot([0,1],[0,1],'--',color='grey',label='Constant-score reference')
    axs[1].axhline(prevalence,ls='--',color='grey',label=f'Prevalence reference {prevalence:.3f}')
    for ax in axs:ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(loc='lower left',fontsize=8)
    axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC across score thresholds')
    axs[1].set(xlabel='Recall',ylabel='Precision',title='Precision recall across score thresholds')
    fig.tight_layout();fig.savefig(out/'07_roc_precision_recall.png',dpi=180);plt.close(fig)
    pd.DataFrame(ap_rows).to_csv(out/'ranking_summary.csv',index=False)
    plot_model_comparison(m,ROOT/'results/model_comparison.png')
    summary={'prepared_rows':len(prepared),'prepared_columns':prepared.columns.tolist(),'development_only_features':features,'duplicates_removed':0,'missing_imputed':0,'note':'Prepared export retains all source records. Target remapped; no one-hot encoding in CSV. Exclude row_id, split and phishing from input features. Added EDA is descriptive; no model or threshold tuning.'}
    (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
    print(conditional.to_string(index=False));print('Joint counts\n',cross);print('Joint phishing rates\n',rates.round(4))
    print('Created',out)
if __name__=='__main__':main()
