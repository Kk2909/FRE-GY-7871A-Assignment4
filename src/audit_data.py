"""Reconcile supplied text datasets and reproduce cross-platform statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, chi2_contingency
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
x = pd.read_csv(DATA / 'twitter_posts_scored.csv', dtype={'tweet_id': str})
n = pd.read_csv(DATA / 'google_news_ai_scored.csv')
r = pd.read_csv(DATA / 'google_news_ai_raw.csv')
m = pd.read_csv(DATA / 'google_news_matched_40_per_day.csv')
qc = pd.read_csv(DATA / 'google_news_daily_qc.csv').set_index('date_ny')
ds = pd.read_csv(DATA / 'google_news_daily_sentiment.csv').set_index('date_ny')
checks = {}
for name, frame in [('x', x), ('news', n), ('matched_news', m)]:
    probs = frame[['prob_negative', 'prob_neutral', 'prob_positive']]
    assert not probs.isna().any().any(), f'{name}: missing probabilities'
    assert np.isfinite(probs).all().all() and probs.ge(0).all().all() and probs.le(1).all().all()
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert np.allclose(frame.sentiment_score, frame.prob_positive-frame.prob_negative, atol=1e-6)
    assert (frame.sentiment_label == probs.idxmax(axis=1).str.replace('prob_', '')).all()
    assert frame.date_ny.between('2026-09-08', '2026-09-30').all()
    checks[name] = {'rows': len(frame), 'days': int(frame.date_ny.nunique()), 'score_consistency': True}
assert len(x) == 920 and x.groupby('date_ny').size().eq(40).all()
assert x.tweet_id.is_unique
assert len(n) == 2764 and n.google_news_url.is_unique and n.headline_normalized.is_unique
assert n.ai_in_title.eq(1).all()
assert len(m) == 920 and m.groupby('date_ny').size().eq(40).all()
assert m.google_news_url.is_unique and m.google_news_url.isin(n.google_news_url).all()
assert n.google_news_url.isin(r.google_news_url).all()
assert len(r[r.ai_in_title.eq(1)].drop_duplicates('headline_normalized')) == len(n)
assert (pd.to_datetime(n.published_utc, utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d') == n.date_ny).all()
assert r.groupby('date_ny').size().equals(qc.n_articles.rename(None))
assert r.groupby('date_ny').source.nunique().equals(qc.n_sources.rename('source'))
assert np.allclose(m.sentiment_score, n.set_index('google_news_url').sentiment_score.reindex(m.google_news_url).values)
nd = n.groupby('date_ny').sentiment_score.mean()
xd = x.groupby('date_ny').sentiment_score.mean()
assert np.allclose(nd, ds.news_mean_sentiment, atol=1e-7)
comparisons = {
    'news_pooled_mean': float(n.sentiment_score.mean()),
    'news_daily_mean': float(nd.mean()),
    'x_daily_mean': float(xd.mean()),
    'daily_mean_difference': float((nd-xd).mean()),
    'daily_paired_mean_p': float(ttest_rel(nd, xd).pvalue),
    'daily_wilcoxon_mean_p': float(wilcoxon(nd-xd).pvalue),
    'matched_news_mean': float(m.sentiment_score.mean()),
    'matched_welch_mean_p': float(ttest_ind(m.sentiment_score, x.sentiment_score, equal_var=False).pvalue),
}
for label in ['negative', 'neutral', 'positive']:
    ns = n.assign(v=n.sentiment_label.eq(label)).groupby('date_ny').v.mean()
    xs = x.assign(v=x.sentiment_label.eq(label)).groupby('date_ny').v.mean()
    assert np.allclose(ns, ds['news_'+label+'_share'], atol=1e-7)
    comparisons[label] = {'news_pooled_share': float(n.sentiment_label.eq(label).mean()), 'x_pooled_share': float(x.sentiment_label.eq(label).mean()), 'news_daily_share': float(ns.mean()), 'x_daily_share': float(xs.mean()), 'paired_p': float(ttest_rel(ns, xs).pvalue)}
nr = n.groupby('date_ny').risk_flag.mean()
xr = x.groupby('date_ny').risk_flag.mean()
assert np.allclose(nr, ds.news_risk_share, atol=1e-7)
comparisons['risk'] = {'news_pooled_share': float(n.risk_flag.mean()), 'x_pooled_share': float(x.risk_flag.mean()), 'news_daily_share': float(nr.mean()), 'x_daily_share': float(xr.mean()), 'paired_p': float(ttest_rel(nr, xr).pvalue)}
ct = pd.DataFrame({'news': m.sentiment_label.value_counts(), 'x': x.sentiment_label.value_counts()}).T
chi, cp, dof, expected = chi2_contingency(ct)
comparisons['matched_chi_square'] = {'statistic': float(chi), 'p': float(cp), 'df': int(dof), 'counts': ct.to_dict()}
checks.update({'raw_rows': len(r), 'raw_sources': int(r.source.nunique()), 'raw_duplicate_urls': int(r.google_news_url.duplicated().sum()), 'x_duplicate_texts': int(x.text.duplicated().sum()), 'news_requested_date_differs_from_publication': int(n.date_ny.ne(n.requested_date).sum())})
(OUT / 'news_comparison.json').write_text(json.dumps({'checks': checks, 'comparisons': comparisons}, indent=2), encoding='utf-8')
nd.rename('news_sentiment').to_frame().join(xd.rename('x_sentiment')).to_csv(OUT / 'daily_sentiment.csv')
print('Data checks passed. Comparisons saved to outputs/news_comparison.json.')
