import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import f as fdist, norm
from types import SimpleNamespace
def grangercausalitytests(a,maxlag=1,verbose=False):
 v=np.asarray(a);y=v[1:,0];xr=np.column_stack([np.ones(len(y)),v[:-1,0]]);xu=np.column_stack([xr,v[:-1,1]])
 er=y-xr@np.linalg.lstsq(xr,y,rcond=None)[0];eu=y-xu@np.linalg.lstsq(xu,y,rcond=None)[0];df=len(y)-3;F=((er@er-eu@eu)/(eu@eu))*df
 return {1:({"ssr_ftest":(F,fdist.sf(F,1,df),df,1)},None)}
def robust_fit(y,x):
 X=np.column_stack([np.ones(len(x)),np.asarray(x)]);Y=np.asarray(y);b=np.linalg.lstsq(X,Y,rcond=None)[0];e=Y-X@b;bread=np.linalg.inv(X.T@X);cov=bread@(X.T@(X*(e**2)[:,None]))@bread*len(Y)/(len(Y)-X.shape[1]);se=np.sqrt(np.diag(cov));names=["const"]+list(x.columns)
 return SimpleNamespace(params=pd.Series(b,index=names),pvalues=pd.Series(2*norm.sf(abs(b/se)),index=names),conf_int=lambda:pd.DataFrame(np.column_stack([b-1.95996398454*se,b+1.95996398454*se]),index=names))
from scipy.stats import linregress, ttest_rel, ttest_ind, wilcoxon, chi2_contingency
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs';out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(root/'data/twitter_posts_scored.csv')
news=pd.read_csv(root/'data/google_news_ai_scored.csv')
matched=pd.read_csv(root/'data/google_news_matched_40_per_day.csv')
news_daily=news.groupby('date_ny').agg(sentiment=('sentiment_score','mean'),risk=('risk_flag','mean'))
daily=d.groupby('date_ny').agg(sentiment=('sentiment_score','mean'),risk=('risk_flag','mean'))
raw=json.load(open(root/'data/market_raw.json'))
prices=pd.DataFrame({k:pd.Series({v['date']:v['adjusted_close'] for v in a['rows']}) for k,a in raw.items()}).sort_index()
ret=prices.pct_change().dropna();tick=['NVDA','MSFT','GOOGL','AMZN','META','AMD','AVGO']
ret['basket']=ret[tick].mean(axis=1);ret['excess']=ret.basket-ret.SPY
sessions=[]
for i,date in enumerate(ret.index):
 prev=ret.index[i-1] if i else '2026-09-04'
 s=daily.loc[(daily.index>=prev)&(daily.index<date)]
 if len(s):sessions.append({'date':date,'prior_sentiment':s.sentiment.mean(),'basket':ret.loc[date,'basket'],'excess':ret.loc[date,'excess']})
z=pd.DataFrame(sessions).set_index('date');z['change']=z.prior_sentiment.diff()
results=[]
for x in ['prior_sentiment','change']:
 for y in ['basket','excess']:
  a=z[[y,x]].dropna();g=grangercausalitytests(a,maxlag=1,verbose=False)[1][0]['ssr_ftest']
  results.append({'x':x,'y':y,'n':len(a),'F':g[0],'p':g[1]})
a=z[['prior_sentiment','basket']].dropna();g=grangercausalitytests(a,maxlag=1,verbose=False)[1][0]['ssr_ftest'];results.append({'x':'basket','y':'prior_sentiment','n':len(a),'F':g[0],'p':g[1]})
# Sign and uncertainty of the primary change specification.
a=z.copy();a['lag_excess']=a.excess.shift();a['lag_change']=a.change.shift();a=a.dropna()
fit=robust_fit(a.excess,a[['lag_excess','lag_change']])
trend=linregress(np.arange(len(daily)),daily.sentiment)
perf=((1+ret[tick+['SPY','basket']]).prod()-1)*100
summary={'news_checks':{'n_raw':3247,'n_scored':len(news),'n_matched':len(matched),'news_pooled_mean':float(news.sentiment_score.mean()),'paired_mean_p':float(ttest_rel(news_daily.sentiment,daily.sentiment).pvalue),'matched_distribution_chi_square':float(chi2_contingency(pd.DataFrame({'news':matched.sentiment_label.value_counts(),'x':d.sentiment_label.value_counts()}).T).statistic)},'tests':results,'performance_percent':perf.to_dict(),'trend_slope':trend.slope,'trend_p':trend.pvalue,'change_coefficient':fit.params.lag_change,'change_robust_p':fit.pvalues.lag_change,'change_CI':fit.conf_int().loc['lag_change'].tolist(),'correlation':ret.basket.corr(ret.SPY)}
(out/'verified_results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.2,2.6));ax.plot(pd.to_datetime(daily.index),daily.sentiment,marker='o',color='#245b82');ax.plot(pd.to_datetime(news_daily.index),news_daily.sentiment,marker='s',color='#e18d35',label='News');ax.lines[0].set_label('X');ax.legend();ax.axhline(0,color='#999',lw=.7);ax.set(ylabel='Mean sentiment',title='Daily AI sentiment: X and news headlines');fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'trend.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,2.5));idx=(1+ret[['basket','SPY']]).cumprod()*100
for k,label in [('basket','AI basket'),('SPY','SPY')]:ax.plot(pd.to_datetime(idx.index),idx[k],label=label)
ax.set(ylabel='Index (Sept 4 close = 100)',title='Daily rebalanced AI basket versus SPY');ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'market.png',dpi=180);plt.close(fig)
from matplotlib.font_manager import findfont, FontProperties
font=findfont(FontProperties(family='DejaVu Sans'));bold=findfont(FontProperties(family='DejaVu Sans',weight='bold'))
pdfmetrics.registerFont(TTFont('DV',font));pdfmetrics.registerFont(TTFont('DVB',bold))
st=getSampleStyleSheet();st.add(ParagraphStyle(name='Body',fontName='DV',fontSize=9.5,leading=14,spaceAfter=8));st.add(ParagraphStyle(name='Head',fontName='DVB',fontSize=14,leading=18,textColor=colors.HexColor('#245b82'),spaceAfter=11));st.add(ParagraphStyle(name='Title2',fontName='DVB',fontSize=19,leading=25,textColor=colors.HexColor('#245b82'),spaceAfter=15));st.add(ParagraphStyle(name='Small2',fontName='DV',fontSize=8,leading=11,spaceAfter=6))
story=[]
def p(s):story.append(Paragraph(s,st['Body']))
def h(s):story.append(Paragraph(s,st['Head']))
def table(rows,widths=None):
 rr=[[Paragraph(str(x),st['Small2']) for x in row] for row in rows];t=Table(rr,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#245b82')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#cccccc'))]));story.append(t);story.append(Spacer(1,10))
def page():story.append(PageBreak())
story.append(Paragraph('AI Risk Sentiment and Equity Returns',st['Title2']));p('Assignment 4 | Khushi Khanna | September 8-30, 2026')
h('1. Question and research design')
p('The assignment begins with Evan Hubinger\'s September 8 statement that advanced AI could threaten humanity. This report asks whether AI sentiment deteriorated over the following 23 days, whether sentiment changes predicted returns in an AI equity basket, and whether news headlines and X posts differed in their coverage of AI risks. The starting statement motivates the window; this design cannot isolate its causal effect without a pre-event comparison or control group.')
p('The verified X sample has 920 posts, with 40 English-language observations per day drawn from two one-hour windows. Searches used “artificial intelligence” or “generative AI.” Every recorded New York date matches its requested collection date. There are no duplicate post IDs; two exact texts recur under different IDs. The sample represents these query terms and time windows, rather than all AI discussion.')
p('Sentiment uses CardiffNLP twitter-roberta-base-sentiment-latest. Each post has negative, neutral and positive probabilities; its continuous score is P(positive) minus P(negative), ranging from -1 to +1. The categorical label is the largest probability. Checks of all 920 rows found no missing scores, inconsistent labels or meaningful probability-sum errors. A risk-language flag measures dictionary matches rather than endorsement of a risk claim.')
p('Souza et al. (2015) link Twitter sentiment with financial variables and apply Granger tests. Following that approach, this report tests predictive relationships in both directions and separates basket returns from returns above a market benchmark. The focus is forecasting content, not structural causation [1].')
p('News collection contains 3,247 raw records from 1,432 named sources, including one repeated URL. Filtering to AI-explicit titles leaves 2,905 records; removing repeated normalized headlines produces 2,764 observations. The final sample contains no repeated URLs or normalized headlines. Publication timestamps place all observations within September 8-30 in New York time. Statistics use publication date rather than the requested search date. The separately supplied matched sample contains exactly 40 headlines per day and 920 observations, all drawn from the final scored file.')
page();h('2. Sentiment over the event window')
p('The X sample is mildly negative on average: -0.0343. Its labels are 29.13% negative, 50.54% neutral and 20.33% positive. The negative fraction exceeds the positive fraction by 8.80 percentage points, but half the sample is neutral. The mean is a classifier-based measure of textual tone, not a direct probability of investor pessimism about AI equities.')
story.append(Image(str(out/'trend.png'),width=480,height=173));story.append(Spacer(1,10))
p('Daily mean sentiment is +0.0560 on September 8 and falls to -0.1668 on September 9, the lowest daily reading. It subsequently fluctuates, including a high of +0.1969 on September 25, and ends at +0.0329 on September 30. The endpoint change is -0.0231, but endpoints alone obscure the intervening reversals.')
p(f'A descriptive linear trend through the 23 daily means has slope {trend.slope:+.4f} score units per day (ordinary least-squares p = {trend.pvalue:.3f}). This does not establish a persistent decline. The trend test assumes independent daily errors and is supplementary rather than a causal event-study test.')
p('The appropriate conclusion is variation without clear sustained deterioration, rather than forcing a downward trend because the initial statement was alarming. Establishing that a specific event drove a given day would require dated article/post examples and comparison with other contemporaneous developments.')
p('Classification has practical limits. Some posts mention AI only incidentally or use sarcasm. Fifty-nine posts exceed 2,000 characters, with a maximum of 17,473; model token limits and truncation should therefore be documented. A human-coded validation sample would help distinguish sentiment toward AI from unrelated emotional language. Retaining the original text and URL alongside model scores permits that review.')
page();h('3. AI basket and observed market performance')
p('The basket contains NVDA, MSFT, GOOGL, AMZN, META, AMD and AVGO. It covers chips, cloud infrastructure and platforms with material AI exposure. Each receives one-seventh weight at the start of every trading day, implying daily rebalancing. SPY serves as the market benchmark. These firms also have substantial non-AI businesses, so their returns cannot be attributed exclusively to AI narratives.')
p('Daily returns are adjusted-close percentage changes. September 8 returns use the September 4 close because September 7 was a market holiday. Cumulative performance therefore covers returns dated September 8-30 and uses the September 4 close as its base. There are 17 return observations. The following results were recomputed from the retained Yahoo Finance price responses.')
table([['Equity / portfolio','Cumulative return']]+[[k,f'{perf[k]:+.2f}%'] for k in tick+['basket','SPY']],[310,170])
story.append(Image(str(out/'market.png'),width=480,height=167));story.append(Spacer(1,8))
p(f'The daily return correlation between the basket and SPY is {ret.basket.corr(ret.SPY):.3f}. Market-adjusted returns are defined as basket return minus SPY return. This is a simple benchmark subtraction, not a beta-adjusted abnormal return. Basket outperformance alongside intermittent negative sentiment is descriptive evidence; it does not establish that sentiment had no effect or that optimism caused gains.')
page();h('4. Granger tests: changes and timing')
p('To prevent same-day look-ahead, all posts dated from the previous trading day through the day before the current session are averaged into a pre-session sentiment value. Friday, Saturday and Sunday observations feed Monday. Posts on September 30 have no subsequent return inside this window and are excluded from predictive tests. September 8 has no earlier sentiment observations, leaving 16 aligned sessions for level tests.')
p('For each session, the change measure is the difference between consecutive pre-session sentiment values. The primary Granger regression predicts current excess return using one lag of excess return and one lag of sentiment change. A restricted regression excludes the sentiment-change term. The F-test asks whether adding that term improves prediction. Level tests are reported as sensitivity checks; the two specifications must not be described interchangeably.')
table([['Predictor → outcome','Input sessions','F','p']]+[[('Sentiment level' if r['x']=='prior_sentiment' else 'Sentiment change' if r['x']=='change' else 'Basket return')+' → '+('basket return' if r['y']=='basket' else 'excess return' if r['y']=='excess' else 'sentiment level'),r['n'],f"{r['F']:.3f}",f"{r['p']:.3f}"] for r in results],[275,75,65,65])
p('“Input sessions” counts aligned rows supplied to the test; a lag-one model loses an additional initial row. Sentiment changes also require one initial difference. The effective regression sample is consequently very small. These recalculated, explicitly aligned results replace the earlier report\'s Granger table and do not reproduce its different, undocumented timing specification.')
p(f'In the primary excess-return model, the coefficient on lagged sentiment change is {fit.params.lag_change:+.4f} return units per one-point score change, with HC1 robust p = {fit.pvalues.lag_change:.3f}. Its 95% robust interval is [{fit.conf_int().loc["lag_change",0]:+.4f}, {fit.conf_int().loc["lag_change",1]:+.4f}]. Direction alone should not be presented as a reliable positive or negative pricing effect when uncertainty includes zero.')
p('One lag limits parameter consumption. Short-window stationarity tests would have little power, and no strong stationarity claim is made here. Differencing sentiment addresses the requested change specification but does not guarantee well-calibrated inference. Multiple exploratory tests, overlapping narratives and omitted firm or macroeconomic events further limit interpretation. Failure to reject a Granger null is not proof of no relationship [1].')
page();h('5. Coverage differences and the COVID comparison')
p('Sacerdote, Sehgal and Cook (2020) compare COVID coverage across sources, topics and changing objective conditions. Their central result is a much larger negative share in major US news than in international news or scientific journals. Their approach motivates comparisons of similar topics and underlying events, rather than treating any difference in negativity as proof of bias [2].')
p('The news probabilities, score arithmetic and labels reconcile for every row. The supplied daily summary matches independently computed daily means. The matched sample reproduces the scored-file values. Pooled news sentiment averages -0.0201, while the equally weighted mean of daily news averages is -0.0268. These differ because headline volume varies across days.')
table([['Pooled measure','News (2,764)','X (920)'],['Negative share','15.52%','29.13%'],['Neutral share','76.05%','50.54%'],['Positive share','8.43%','20.33%'],['Risk-language share','17.51%','25.00%']],[200,140,140])
p('For the 23 paired days, the news-minus-X mean sentiment difference is +0.0075: paired t-test p = 0.6750 and Wilcoxon p = 0.7090. Equal weighting of days gives negative shares of 16.02% versus 29.13% (p &lt; 0.0001), positive shares of 8.38% versus 20.33% (p &lt; 0.0001), and risk-language shares of 17.44% versus 25.00% (p = 0.0000865). Thus, the most visible difference is in composition, not average tone.')
p('The 920-headline matched sample has mean sentiment -0.0297 versus -0.0343 for X (Welch p = 0.8364). Its category distribution differs strongly from X: chi-square = 119.37, two degrees of freedom, p &lt; 0.0001. This test assumes independent observations; common sources, authors and events may create clustering. Daily paired comparisons partly address within-day dependence, but neither comparison identifies a platform effect.')
p('Risk headlines cover diverse issues: Reuters on September 8 discusses the OpenAI/New York Times copyright case; CBS News on September 9 reports the researcher’s human-extinction warning; Computerworld on September 9 discusses evidence against universal AI-driven job losses. These examples are located in the supplied headline file by source and publication date. A dictionary hit therefore does not necessarily mean alarmist coverage or endorsement of danger.')
p('Compared with COVID, this sample does not show uniquely greater professional-news negativity. X has larger shares at both sentiment extremes and more risk vocabulary. Longer posts have more opportunities for dictionary matches than headlines. Topic mix, model calibration, repeated authors, syndicated variants and sarcasm remain alternative explanations. Google News searches also cap results, and named sources are not all professional newsrooms. The defensible conclusion is a sampled coverage difference; topic matching, human labels, source grouping and length controls are needed to establish systematic bias.')
page();h('6. Conclusions and submission evidence')
p('The verified X sample shows mildly negative average tone, frequent daily reversals and no clear sustained deterioration. A daily rebalanced AI basket rose over the window despite those fluctuations. The Granger analysis explicitly distinguishes sentiment changes from levels and uses only information available before the predicted trading session. Its small sample supports cautious inference rather than a strong causal claim.')
p('The verified news results show more neutral headlines and larger negative and positive shares on X. Mean sentiment is statistically similar across channels, while sentiment composition and risk vocabulary differ significantly in the sample. These are coverage differences, not sufficient evidence of systematic bias. Following the COVID study, stronger bias analysis requires topic and event controls. The 23-day window, fixed X search windows, Google News result caps, broad queries, model errors and possible truncation limit generalization.')
p('Supporting evidence includes the 920 scored X posts, raw and scored news files, the matched news sample, daily news quality checks and sentiment summaries, retained market prices and analysis code. Text-level score consistency, publication-date validity, sample membership, daily summaries and the reported cross-platform tests were checked. This verifies calculations on supplied records; it does not replace manual sentiment validation or an independent audit of search coverage.')
h('References and data sources')
p('[1] Souza, T. T. P., Kolchyna, O., Treleaven, P. C., and Aste, T. (2015). <i>Twitter Sentiment Analysis Applied to Finance: A Case Study in the Retail Industry.</i> arXiv:1507.00784. Class reading. https://arxiv.org/abs/1507.00784')
p('[2] Sacerdote, B., Sehgal, R., and Cook, M. (2020). <i>Why Is All COVID-19 News Bad News?</i> NBER Working Paper 28110, November. Class reading. https://www.nber.org/papers/w28110')
p('[3] CardiffNLP. <i>twitter-roberta-base-sentiment-latest.</i> Sentiment model used in the supplied scored-post file. https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest')
p('[4] X post search sample, September 8-30, 2026: <i>twitter_posts_scored.csv</i>, 920 rows. Original post URLs are preserved in the data. Assignment-provided Hubinger statement supplies event context.')
p('[5] Yahoo Finance chart responses for NVDA, MSFT, GOOGL, AMZN, META, AMD, AVGO and SPY, retained locally with retrieval timestamps; adjusted closes September 4-30, 2026. Returns and tests recomputed for this revision.')
p('[6] Google News RSS/search dataset, September 8-30, 2026. Supplied files: google_news_ai_raw.csv; google_news_ai_scored.csv; google_news_matched_40_per_day.csv; google_news_daily_qc.csv; google_news_daily_sentiment.csv. Each headline retains its source, publication timestamp and Google News URL. Cross-platform statistics independently recomputed from these files.')
def footer(c,doc):
 c.setFont('DV',8);c.setFillColor(colors.HexColor('#555555'));c.drawString(48,27,'Khushi Khanna | AI sentiment analysis');c.drawRightString(564,27,f'{doc.page} / 6')
SimpleDocTemplate(str(out/'AI_Risk_Sentiment_Assignment4.pdf'),pagesize=(612,792),rightMargin=48,leftMargin=48,topMargin=44,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
