#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import pandas as pd

WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)
DASH_RE = re.compile(r"[\u2010-\u2015\-]+")

def normalize_ws(v):
    if v is None or pd.isna(v): return ""
    return re.sub(r"\s+", " ", str(v)).strip()

def tokenize(text: str) -> List[str]:
    text = DASH_RE.sub(" ", normalize_ws(text).lower())
    return WORD_RE.findall(text)

def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))

def column_letter_to_index(letter: str) -> int:
    idx = 0
    for ch in letter.strip().upper(): idx = idx*26 + (ord(ch)-64)
    return idx-1

def resolve_columns(cols: List[str], specs: Iterable[str]) -> List[str]:
    out=[]
    for raw in specs:
        spec=raw.strip()
        if not spec: continue
        if spec in cols: out.append(spec); continue
        if re.fullmatch(r"\d+", spec):
            i=int(spec)
            if 0 <= i < len(cols): out.append(cols[i])
            elif 1 <= i <= len(cols): out.append(cols[i-1])
            else: raise ValueError(f"Column index out of range: {spec}")
            continue
        if re.fullmatch(r"[A-Za-z]+", spec):
            i=column_letter_to_index(spec)
            if 0 <= i < len(cols): out.append(cols[i])
            else: raise ValueError(f"Column letter out of range: {spec}")
            continue
        raise ValueError(f"Could not resolve column: {spec}")
    return list(dict.fromkeys(out))

def split_terms_blob(blob: str) -> List[str]:
    return [p.strip().strip('"\'') for p in re.split(r"[,;|]", blob) if p.strip().strip('"\'')]

def load_dictionary(path: Path):
    lines=[ln.strip() for ln in path.read_text(encoding='utf-8-sig',errors='replace').splitlines() if ln.strip() and not ln.strip().startswith('#')]
    if not lines: raise ValueError('Dictionary has no usable lines')
    entries=[]
    first=[x.strip().lower() for x in re.split(r"[,\t]", lines[0])]
    if 'term' in first or 'terms' in first or 'topic' in first:
        dialect=csv.Sniffer().sniff('\n'.join(lines[:10]), delimiters=',\t')
        for row in csv.DictReader(lines, dialect=dialect):
            lc={str(k).strip().lower(): normalize_ws(v) for k,v in row.items() if k is not None}
            topic=lc.get('topic') or lc.get('category') or lc.get('theme') or 'dictionary'
            blob=lc.get('term') or lc.get('terms') or lc.get('keyword') or lc.get('keywords') or ''
            for term in split_terms_blob(blob): entries.append((topic,term))
    else:
        for line in lines:
            if ':' in line:
                topic, blob = line.split(':',1)
                for term in split_terms_blob(blob): entries.append((normalize_ws(topic) or 'dictionary', term))
            elif '\t' in line:
                topic, term = line.split('\t',1); entries.append((normalize_ws(topic) or 'dictionary', normalize_ws(term)))
            else:
                entries.append(('dictionary', normalize_ws(line)))
    seen=set(); clean=[]
    for topic,term in entries:
        toks=tuple(tokenize(term))
        if not toks: continue
        key=(topic.lower(), toks)
        if key in seen: continue
        seen.add(key); clean.append({'topic':topic, 'term':term, 'tokens':toks})
    if not clean: raise ValueError('No usable dictionary terms')
    return clean

def build_term_index(entries):
    # Token trie: each node is a dict. Terminal entries are stored under '_terms'.
    root = {}
    max_len = 1
    for e in entries:
        toks = e['tokens']
        max_len = max(max_len, len(toks))
        node = root
        for tok in toks:
            node = node.setdefault(tok, {})
        node.setdefault('_terms', []).append((toks, e['topic'], e['term']))
    return root, max_len

def make_context(text, term_tokens, window):
    if not text or not term_tokens: return ''
    pat = re.compile(r"(?<![A-Za-z0-9_])" + r"[\s\u2010-\u2015\-]+".join(map(re.escape, term_tokens)) + r"(?![A-Za-z0-9_])", re.I)
    m=pat.search(text)
    if not m: return ''
    a=max(0,m.start()-window); b=min(len(text),m.end()+window)
    return ('...' if a else '')+normalize_ws(text[a:b])+('...' if b<len(text) else '')

def score_text(text, trie_root, max_len, article_words, mode, window):
    toks=tokenize(text)
    phrase_counts=Counter()
    n=len(toks)
    for i in range(n):
        node=trie_root
        j=i
        while j<n and toks[j] in node:
            node=node[toks[j]]
            if '_terms' in node:
                for term_tokens, _topic, _term in node['_terms']:
                    phrase_counts[term_tokens]+=1
            j+=1
    term_counts=Counter(); topic_counts=Counter(); topic_unique=defaultdict(set)
    # Retrieve terminal entries by walking the trie for each matched phrase.
    for phrase,count in phrase_counts.items():
        node=trie_root
        for tok in phrase:
            node=node[tok]
        for _tokens, topic, term in node.get('_terms', []):
            term_counts[term]+=count; topic_counts[topic]+=count; topic_unique[topic].add(term.lower())
    total=int(sum(term_counts.values())); unique=int(len(term_counts)); topics=int(len(topic_counts))
    density=round((total/article_words)*1000,4) if article_words else 0.0
    if mode=='total_matches': raw=float(total); basis='total_dictionary_term_mentions'
    elif mode=='density': raw=float(density); basis='dictionary_term_mentions_per_1000_words'
    elif mode=='hybrid': raw=float(unique + math.log1p(total) + .05*density); basis='unique_terms_plus_log_repeated_mentions_plus_density_adjustment'
    else: raw=float(unique); basis='unique_dictionary_terms_matched'
    top=topic_counts.most_common(1)[0][0] if topic_counts else ''
    first_phrase=next(iter(phrase_counts), tuple())
    return {
        'dict_match_count_total': total,
        'dict_unique_terms_count': unique,
        'dict_topic_count': topics,
        'match_density_per_1000_words': density,
        'coherence_raw_score': round(raw,6),
        'coherence_raw_score_basis': basis,
        'story_fit_topic': top,
        'matched_terms': '; '.join(f'{t} ({c})' for t,c in term_counts.most_common()),
        'matched_topics': '; '.join(f'{t} ({c})' for t,c in topic_counts.most_common()),
        'topic_match_counts_json': json.dumps(dict(topic_counts), ensure_ascii=False),
        'matched_terms_by_topic_json': json.dumps({k:sorted(v) for k,v in topic_unique.items()}, ensure_ascii=False),
        'story_fit_sample_text': make_context(text, first_phrase, window),
    }

def coherence_bin(score):
    score=float(score or 0)
    if score>=100: return '100'
    lo=int(score//10)*10
    return f'{lo:02d}-{lo+10:02d}'

def distribution(df, col):
    bins=[f'{i:02d}-{i+10:02d}' for i in range(0,100,10)]+['100']
    d=df[col].map(coherence_bin).value_counts().reindex(bins, fill_value=0).rename_axis('coherence_bin').reset_index(name='article_count')
    total=int(d.article_count.sum())
    d['article_pct']=(d.article_count/total*100).round(2) if total else 0
    d['cumulative_article_count']=d.article_count.cumsum()
    d['cumulative_article_pct']=(d.cumulative_article_count/total*100).round(2) if total else 0
    return d

def add_relative(df, threshold):
    df=df.copy(); df['story_fit']=(df.dict_match_count_total>0).astype(int)
    matched=df.story_fit==1
    df['coherence_percentile']=0.0; df['coherence_score']=0.0
    if matched.any():
        pct=df.loc[matched,'coherence_raw_score'].rank(method='average', pct=True)*100
        df.loc[matched,'coherence_percentile']=pct.round(2); df.loc[matched,'coherence_score']=pct.round(2)
    df['coherence_bin']=df.coherence_score.map(coherence_bin)
    df['high_coherence']=(df.coherence_score>=threshold).astype(int)
    high=df.high_coherence==1
    summary={
        'article_count_cleaned': int(len(df)),
        'dictionary_matching_article_count': int(matched.sum()),
        'dictionary_matching_article_pct': round(float(matched.mean()*100),2) if len(df) else 0,
        'relative_coherence_threshold': threshold,
        'high_coherence_count': int(high.sum()),
        'high_coherence_pct_of_cleaned': round(float(high.mean()*100),2) if len(df) else 0,
        'high_coherence_pct_of_matching': round(float(high.sum()/matched.sum()*100),2) if matched.sum() else 0,
        'raw_score_min_for_high_coherence': float(df.loc[high,'coherence_raw_score'].min()) if high.any() else None,
        'coherence_score_definition':'Percentile rank from 0-100 among dictionary-matching articles; non-matching articles receive 0.',
        'raw_score_basis': str(df.coherence_raw_score_basis.iloc[0]) if len(df) else '',
        'raw_score_p50_matching': round(float(df.loc[matched,'coherence_raw_score'].quantile(.5)),4) if matched.any() else 0,
        'raw_score_p80_matching': round(float(df.loc[matched,'coherence_raw_score'].quantile(.8)),4) if matched.any() else 0,
        'raw_score_p90_matching': round(float(df.loc[matched,'coherence_raw_score'].quantile(.9)),4) if matched.any() else 0,
        'raw_score_max_matching': round(float(df.loc[matched,'coherence_raw_score'].max()),4) if matched.any() else 0,
    }
    return df, summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input_csv'); ap.add_argument('--dictionary', default='dictionary.txt'); ap.add_argument('--outdir', default='preprocessed_articles_relative')
    ap.add_argument('--article-col', default='article_text'); ap.add_argument('--required-cols', default='A,B,C,D')
    ap.add_argument('--score-text-cols', default='article_text,headline'); ap.add_argument('--min-word-count', type=int, default=25)
    ap.add_argument('--raw-score-mode', choices=['unique_terms','total_matches','density','hybrid'], default='unique_terms')
    ap.add_argument('--coherence-threshold', type=float, default=80); ap.add_argument('--context-window', type=int, default=180); ap.add_argument('--max-rows', type=int)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    entries=load_dictionary(Path(args.dictionary)); term_index,max_len=build_term_index(entries)
    df=pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if args.max_rows: df=df.head(args.max_rows).copy()
    cols=list(df.columns); required=resolve_columns(cols,args.required_cols.split(','))
    if args.article_col not in df.columns: raise ValueError(f"Article column '{args.article_col}' not found. Available: {cols}")
    score_cols=[c for c in resolve_columns(cols,args.score_text_cols.split(',')) if c in df.columns] or [args.article_col]
    for c in df.columns: df[c]=df[c].map(normalize_ws)
    input_rows=len(df); reqmiss=df[required].eq('').any(axis=1); artmiss=df[args.article_col].eq(''); wc=df[args.article_col].map(word_count); short=wc<args.min_word_count
    keep=~reqmiss & ~artmiss & ~short
    removed={'input_rows':int(input_rows),'removed_missing_required_A_D':int(reqmiss.sum()),'removed_missing_article_text':int((~reqmiss & artmiss).sum()),'removed_article_under_min_words':int((~reqmiss & ~artmiss & short).sum()),'kept_after_cleaning':int(keep.sum())}
    clean=df.loc[keep].copy(); clean['article_word_count']=clean[args.article_col].map(word_count)
    texts=clean[score_cols].agg(' '.join, axis=1)
    records=[score_text(t,term_index,max_len,int(w),args.raw_score_mode,args.context_window) for t,w in zip(texts, clean.article_word_count)]
    clean=pd.concat([clean,pd.DataFrame(records,index=clean.index)],axis=1)
    clean,summary=add_relative(clean,args.coherence_threshold)
    paths={
        'output_csv':out/'articles_cleaned_scored.csv',
        'story_fit_only_csv':out/'articles_story_fit_only.csv',
        'high_coherence_csv':out/f'articles_coherence_relative_{int(args.coherence_threshold)}plus.csv',
        'coherence_summary_csv':out/'coherence_summary.csv',
        'coherence_distribution_csv':out/'coherence_distribution.csv',
        'raw_score_distribution_csv':out/'raw_score_distribution.csv',
        'coherence_by_topic_csv':out/'coherence_by_topic.csv',
        'removed_row_counts_csv':out/'removed_row_counts.csv',
        'audit_json':out/'preprocess_audit_summary.json'}
    clean.to_csv(paths['output_csv'],index=False)
    clean[clean.story_fit==1].to_csv(paths['story_fit_only_csv'],index=False)
    clean[clean.high_coherence==1].to_csv(paths['high_coherence_csv'],index=False)
    pd.DataFrame([summary]).to_csv(paths['coherence_summary_csv'],index=False)
    distribution(clean,'coherence_score').to_csv(paths['coherence_distribution_csv'],index=False)
    raw=clean.loc[clean.story_fit==1,'coherence_raw_score'].value_counts().sort_index().rename_axis('coherence_raw_score').reset_index(name='matching_article_count')
    raw['matching_article_pct']=(raw.matching_article_count/raw.matching_article_count.sum()*100).round(2) if len(raw) else 0
    raw.to_csv(paths['raw_score_distribution_csv'],index=False)
    topics=[]
    for topic,sub in clean[clean.story_fit==1].groupby('story_fit_topic'):
        topics.append({'story_fit_topic':topic,'matching_article_count':int(len(sub)),'mean_coherence_score':round(float(sub.coherence_score.mean()),2),'median_coherence_score':round(float(sub.coherence_score.median()),2),'high_coherence_count':int(sub.high_coherence.sum()),'mean_unique_terms':round(float(sub.dict_unique_terms_count.mean()),2),'median_unique_terms':round(float(sub.dict_unique_terms_count.median()),2)})
    pd.DataFrame(topics).sort_values('matching_article_count', ascending=False).to_csv(paths['coherence_by_topic_csv'],index=False) if topics else pd.DataFrame().to_csv(paths['coherence_by_topic_csv'],index=False)
    pd.DataFrame([removed]).to_csv(paths['removed_row_counts_csv'],index=False)
    audit={'input_csv':args.input_csv,'dictionary':args.dictionary,'required_columns':required,'article_column':args.article_col,'score_text_columns':score_cols,'min_word_count':args.min_word_count,'raw_score_mode':args.raw_score_mode,'coherence_threshold':args.coherence_threshold,'dictionary_terms_loaded':len(entries),'max_dictionary_phrase_words':max_len,'counts':removed,'summary':summary, **{k:str(v) for k,v in paths.items()}}
    paths['audit_json'].write_text(json.dumps(audit,indent=2,ensure_ascii=False), encoding='utf-8')
    print(json.dumps(audit,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
