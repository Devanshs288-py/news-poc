# Personalized News Recommendation POC

This proof of concept tests whether semantic retrieval, an interpretable hybrid
score, and Cohere reranking produce better personalized news recommendations
than keyword matching alone.

The implementation is contained in
[`personalized_news_poc.ipynb`](personalized_news_poc.ipynb). It reads CRM data
from [`CRM_Input_template.xlsx`](CRM_Input_template.xlsx), discovers recent news,
scores every article for every contact, and exports recommendation and human
evaluation workbooks.

## GPT web-search experiment

[`personalized_news_gpt_web_search_poc.ipynb`](personalized_news_gpt_web_search_poc.ipynb)
is a separate experiment that replaces Tavily with GPT web search through the
OpenAI Responses API and Portkey. It makes one cached search call per normalized
interest, retrieves ten articles, and selects seven with a BM25/embedding
hybrid score. Background response IDs are cached immediately so timed-out or
interrupted searches can resume without creating another request. It does not
use the Cohere reranker.

[`personalized_news_trending_engagement_poc.ipynb`](personalized_news_trending_engagement_poc.ipynb)
keeps that pipeline unchanged but uses an engagement-focused search brief. It
prioritizes recent, prominent, widely covered stories that are likely to be
recognized and discussed by clients, then exports its own workbook and cache
under `gpt_web_search_trending_engagement_output/`. Because comparable article
view counts are rarely public, it asks web search to use observable attention
signals instead of inventing a precise most-viewed ranking.

Set `PORTKEY_LLM_MODEL` in `.env` before running its web-search section. All of
its generated workbooks and caches are isolated under
[`gpt_web_search_output/`](gpt_web_search_output/README.md).

## What the notebook is trying to prove

The experiment compares three ways of matching news to a contact:

1. **Keyword matching** — overlap between profile/account words and article
   words.
2. **Semantic matching** — cosine similarity between OpenAI embeddings of a
   contact profile and an article.
3. **Hybrid retrieval plus reranking** — semantic, entity, keyword, recency,
   source-quality, and account signals are combined, after which Cohere judges
   whether each leading article is genuinely useful as a reason to contact the
   person.

The POC is successful only if blinded human reviewers rate the hybrid/reranked
results higher than the keyword and semantic baselines. Higher model-generated
scores alone are not treated as proof of improvement.

Two decisions are deliberately kept separate:

- **Article relevance:** what news should be recommended to a contact?
- **Outreach priority:** which contacts should be contacted now?

The outreach-priority score never affects article retrieval or ranking. This
prevents CRM urgency from making an irrelevant article appear relevant.

## End-to-end flow

```text
CRM workbook
    |
    +-- contact interests and role --------+
    +-- account context -------------------+--> search queries
    |                                      |        |
    +-- tier/relationship/last engagement  |        v
              |                            |   Tavily news search
              v                            |        |
       outreach-priority score             |        v
       (kept separate)                     |   deduplicated articles
                                           |        |
                                           v        v
                                      OpenAI embeddings
                                               |
                                               v
                                     hybrid candidate score
                                               |
                                      top 20 per contact and
                                             category
                                               |
                                               v
                                        Cohere reranking
                                               |
                                               v
                              3 professional + 2 personal articles
                                               |
                          +--------------------+--------------------+
                          v                                         v
              recommendation workbook                   blinded evaluation sheet
```

## Repository files

| Path | Purpose |
| --- | --- |
| `personalized_news_poc.ipynb` | Complete data preparation, retrieval, scoring, reranking, export, and evaluation workflow. |
| `personalized_news_gpt_web_search_poc.ipynb` | Separate GPT web-search, deduplicated-interest, BM25/semantic experiment. |
| `personalized_news_trending_engagement_poc.ipynb` | Engagement-focused GPT web-search variant with isolated cache, workbook, and final cost summary. |
| `llm_service.py` | Portkey-backed text generation and Responses API web-search helpers. |
| `CRM_Input_template.xlsx` | Source CRM workbook containing contacts, accounts, interests, tiers, and engagement data. |
| `poc_cache/` | Generated news, embedding, and reranking caches. It is ignored by Git. |
| `gpt_web_search_output/` | Generated GPT web-search workbook and per-interest caches. |
| `personalized_news_recommendations.xlsx` | Generated final recommendations and separate outreach-priority list. |
| `personalized_news_evaluation.xlsx` | Generated blinded sheet for human relevance ratings. |

The two output workbooks are created only after their corresponding notebook
cells run successfully. They are ignored by Git.

## Requirements and setup

The notebook metadata uses Python 3.12. It installs these runtime packages in
its first code cell:

- `pandas`
- `numpy`
- `scikit-learn`
- `openpyxl`
- `openai`
- `tavily-python`
- `cohere`

Open the notebook in JupyterLab, Jupyter Notebook, or VS Code and select the
project's Python environment. If creating a new environment manually:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter pandas numpy scikit-learn openpyxl openai tavily-python cohere
```

The following credentials must be present in the notebook kernel's environment:

```bash
export TAVILY_API_KEY="..."
export OPENAI_API_KEY="..."
export COHERE_API_KEY="..."
```

The notebook reads variables with `os.getenv`; it does **not** load `.env`
automatically. If credentials are stored in the repository's ignored `.env`
file, export them before starting Jupyter, for example:

```bash
set -a
source .env
set +a
jupyter lab
```

The current `.env` template includes Portkey-related settings, but the current
notebook constructs a standard `OpenAI` client and specifically requires
`OPENAI_API_KEY`. Portkey settings are not used by this notebook unless the
embedding client code is changed.

Never commit API keys, `.env`, generated caches, or CRM-derived output files.

## Important configuration

The main controls are defined near the top of the notebook:

| Setting | Current value | Meaning |
| --- | ---: | --- |
| `LOOKBACK_DAYS` | `30` | News retrieval window ending on the run date (UTC). |
| `TAVILY_MAX_RESULTS` | `10` | Maximum results requested for each search query. |
| `TAVILY_SEARCH_DEPTH` | `advanced` | Tavily search mode. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model used for profile and article vectors. |
| `RERANK_MODEL` | `rerank-v4.0-pro` | Cohere model used to rerank candidates. |
| `RERANK_CANDIDATES` | `20` | Highest hybrid-scoring articles reranked per contact/category. |
| `FINAL_RERANK_WEIGHT` | `0.65` | Contribution of the Cohere score to the final score. |
| `REFRESH_NEWS` | `False` | Reuse the matching dated news cache when available. |

Changing models or scoring weights changes the experiment and may invalidate a
direct comparison with earlier human ratings.

## Detailed notebook walkthrough

### 1. Inspect and minimize CRM data

The notebook inventories every sheet in the input workbook, but imports data
from only three sheets:

- `Input Contact Information` supplies company, contact name, designation,
  seniority, relationship quality, automatic/manual tier, professional and
  personal interests, and last engagement date.
- `Accounts Overview` supplies sector, products/services, competitors, and
  headquarters.
- `Methodology` is displayed only so the tier and cadence assumptions remain
  auditable.

Calculated overview sheets, navigation sheets, email addresses, phone numbers,
LinkedIn URLs, communication channels, owners/guardians, and helper columns are
not loaded. This is an intentional data-minimization measure.

The cleaning stage:

- renames workbook headings to consistent internal names;
- removes records without a contact or company;
- deduplicates contacts by `(contact, company)` and accounts by company;
- normalizes whitespace and parses engagement dates;
- uses the manual tier when present, otherwise the automatic tier;
- performs a many-to-one contact-to-account merge;
- fails with an assertion if any contact lacks account context; and
- creates stable in-run contact IDs such as `C0001`.

Three text profiles are then built:

- `professional_profile`: designation plus professional interests;
- `personal_profile`: personal interests only; and
- `account_context`: company, sector, products/services, competitors, and
  geography.

With the current workbook, the prepared dataset contains 45 contacts across 16
accounts. These counts will change when the source workbook changes.

### 2. Calculate outreach priority separately

The notebook creates an operational priority score from structured CRM fields.
It does not use article content or article-relevance scores.

Cadence and review thresholds are:

| Tier | Target cadence | Review threshold |
| --- | ---: | ---: |
| Tier 1 | 30 days | 60 days |
| Tier 2 | 90 days | 180 days |
| Tier 3 | 180 days | 365 days |

`days_since_engagement` is calculated relative to the UTC run date.
`engagement_urgency` is `1.0` when no engagement date exists; otherwise it is:

```text
clip(days_since_engagement / target_cadence_days, 0, 2) / 2
```

This means urgency rises linearly from 0 to 1 and reaches its maximum at twice
the target cadence. Engagement status is assigned as follows:

- no date: `No Task`;
- within target cadence: `On Track`;
- after cadence but within review threshold: `For Review`;
- after review threshold: `Overdue`.

Seniority, tier, and relationship quality are mapped to numeric values:
`High`/`Tier 1` = 1.0, `Medium`/`Tier 2` = 0.6, and `Low`/`Tier 3` = 0.2.
The final 0–100 priority score is:

```text
100 * (
    0.25 * seniority
  + 0.25 * tier
  + 0.20 * relationship_quality
  + 0.30 * engagement_urgency
)
```

### 3. Build batched news-search queries

Interest strings are split on semicolons. For text such as
`Energy transition — interested in ...`, only the short label before the dash
(`Energy transition`) becomes a search theme.

The notebook creates:

- one account query per company, using company, sector, products/services, and
  up to three competitors;
- one professional-theme query per company, using up to eight distinct themes
  aggregated across that company's contacts; and
- one query for each distinct personal-interest theme across all contacts.

This batching improves coverage without making a separate API request for every
contact-interest pair. The current workbook generates 16 account, 16
professional, and 22 personal queries (54 total).

### 4. Retrieve, normalize, deduplicate, and cache news

For each query, Tavily is asked for dated news within the configured 30-day UTC
window. The response retains the title, URL, source domain, publication date,
summary/content, discovery query, and discovery type.

URLs are canonicalized by:

- lowercasing the scheme and host;
- removing fragments and a trailing slash;
- removing `utm_*`, `fbclid`, `gclid`, `mc_cid`, and `mc_eid` tracking
  parameters; and
- retaining other query parameters.

Results with the same canonical URL are grouped into one article. If a URL is
empty, a normalized title is used as the fallback deduplication key. Discovery
queries/types that found the same article are joined so provenance is not lost.
Rows without both a title and content are removed, article IDs such as `A00001`
are assigned, and the embedding text becomes:

```text
Title: <title>. Summary: <content>
```

The news cache name contains the start date, end date, and a SHA-256-derived
hash of the query set and window. With `REFRESH_NEWS = False`, an exact cache
match avoids new Tavily requests. Set it to `True` to force retrieval.

### 5. Generate and cache embeddings

OpenAI embeddings are generated for exactly four kinds of text:

- professional profiles;
- personal profiles;
- account contexts; and
- article title/summary text.

Operational fields such as tier, relationship quality, and last engagement date
are never sent for embedding. Texts are normalized, deduplicated, and sent in
batches of up to 100.

Each cached vector is keyed by a SHA-256 hash of the embedding model name and
the exact input text. This allows repeated runs to request only missing vectors
while preventing reuse across different embedding models.

### 6. Score every contact/category/article combination

Every article is scored twice for each contact: once as a `professional`
candidate and once as a `personal` candidate. Professional and personal
recommendations are ranked independently.

The score components are:

- `semantic_similarity`: cosine similarity between the relevant interest
  profile and article embedding;
- `account_context_similarity`: cosine similarity between account context and
  the article;
- `entity_company_score`: 1.0 when the article names the contact's company,
  0.6 when it names a listed competitor, otherwise 0;
- `keyword_score`: normalized overlap of non-stopword profile/account tokens
  with article tokens, capped at 1.0;
- `recency_score`: linear decay from 1 to 0 across the 30-day lookback window;
  and
- `source_quality_score`: a configured value for selected publishers, with
  0.50 as the default for an unlisted domain.

Cosine similarities remain available as raw diagnostic columns but are clipped
to the 0–1 range before entering the weighted score.

#### Professional hybrid score

| Component | Weight |
| --- | ---: |
| Professional-interest semantic similarity | 40% |
| Account-context semantic similarity | 20% |
| Company/competitor entity evidence | 15% |
| Keyword overlap | 10% |
| Recency | 10% |
| Source quality | 5% |

#### Personal hybrid score

| Component | Weight |
| --- | ---: |
| Personal-interest semantic similarity | 60% |
| Keyword overlap | 20% |
| Recency | 15% |
| Source quality | 5% |
| Account context | 0% |
| Company/competitor entity evidence | 0% |

The zero account and entity weights are important: a personal recommendation
cannot rank higher because of the contact's job or company.

### 7. Rerank the leading candidates with Cohere

For each `(contact, category)` pair, only the 20 highest hybrid-scoring articles
are sent to Cohere. Each reranker document contains the article title, source,
publication date, and up to 2,500 characters of summary content.

For professional candidates, the query includes the person's role,
professional interests, and account context. For personal candidates, it
explicitly includes only personal interests and instructs the model not to use
professional or company context. The prompt asks whether the story is a useful,
genuine reason for a relationship manager or consultant to contact the person,
not merely whether keywords overlap.

The final score is:

```text
final_score = 0.65 * reranker_score + 0.35 * hybrid_score
```

Rerank responses are cached using a hash of the reranker model, query, and exact
candidate documents.

### 8. Select and explain recommendations

The notebook selects, per contact:

- the top 3 professional articles; and
- the top 2 personal articles.

It assigns ranks within each category and generates a readable explanation. A
professional explanation identifies matched interests and whether the article
mentions the company, a competitor, or only fits the sector context. A personal
explanation identifies matching personal interests and states that the choice
was independent of CRM urgency and professional context.

The resulting recommendation table includes:

- contact and company;
- article title, publisher, date, and URL;
- professional/personal category;
- semantic, account, entity, keyword, recency, and hybrid scores;
- Cohere reranker score and final rank; and
- `why_this_is_relevant` explanation.

An assertion verifies that every contact has exactly three professional and two
personal recommendations.

### 9. Export recommendation and outreach workbooks

`personalized_news_recommendations.xlsx` contains two sheets:

- `Recommendations`: the five selected articles per contact and their scoring
  evidence; and
- `Outreach Priority`: the independently calculated CRM priority list.

Keeping these on separate sheets makes it possible to decide both whom to
contact and what relevant story to discuss without mixing the two scores.

### 10. Inspect one contact across methods

`show_contact_results(contact_name)` displays the selected contact's profile and
then shows, separately for professional and personal categories:

- keyword top 5;
- semantic top 5;
- hybrid top 5;
- Cohere-reranked top 5; and
- final recommendations.

Name matching is case-insensitive. If an exact name is not found, the function
raises an error containing possible partial matches. Change the last notebook
line to inspect another person:

```python
show_contact_results("Contact Name")
```

### 11. Create a blinded human-evaluation workbook

The evaluation export takes the top five results for every contact and category
from three methods:

- `Keyword` by `keyword_score`;
- `Semantic` by `semantic_similarity`; and
- `Hybrid/Rerank` by `final_score`.

It exports article identity and rank but not model scores, reducing the chance
that reviewers are influenced by the model's confidence. Reviewers fill in:

- `human_relevance_score`: 0 = irrelevant, 1 = weakly related, 2 = relevant,
  3 = excellent reason to reach out;
- `human_outreach_appropriate`; and
- `human_comments`.

With 45 contacts, two categories, three methods, and five results per method,
the current evaluation design produces 1,350 rows.

After reviewers complete the workbook, run:

```python
display(evaluate_human_scores())
```

The function validates that scores are blank or between 0 and 3, then reports:

- average human relevance;
- number of rated articles;
- mean Precision@3, treating ratings 2 and 3 as relevant;
- percentage of ratings equal to 3; and
- percentage of ratings equal to 0.

Precision@3 for a contact/category is left undefined until all three leading
results in that group have been rated.

## How to run the experiment

1. Populate `CRM_Input_template.xlsx` without renaming the required sheets or
   columns.
2. Activate the Python environment and expose `TAVILY_API_KEY`,
   `OPENAI_API_KEY`, and `COHERE_API_KEY` to the notebook kernel.
3. Open `personalized_news_poc.ipynb` and run cells from top to bottom.
4. Review the generated `personalized_news_recommendations.xlsx`.
5. Give `personalized_news_evaluation.xlsx` to human reviewers without exposing
   the model-score tables.
6. Place the completed evaluation file at the same path and run the final
   metrics cell.
7. Compare `Hybrid/Rerank` against both baselines using Precision@3 and average
   human relevance.

When input data, dates, query construction, weights, or models change, generate
a new evaluation workbook instead of mixing ratings from different experiment
versions.

## Cache behavior

The notebook may create these files under `poc_cache/`:

- `articles_<start>_<end>_<hash>.json`: deduplicated Tavily results;
- `embeddings.json`: vectors keyed by model and exact text; and
- `reranks.json`: Cohere results keyed by model, query, and candidates.

Caches reduce cost and make repeated analysis faster. To fetch news again for
the same window, set `REFRESH_NEWS = True`. To fully recompute another stage,
move or remove only its specific cache file while the notebook is stopped.
Cache files can contain CRM-derived text and article content, so treat them as
sensitive working data even though they are Git-ignored.

## Troubleshooting

### Missing environment variable

An error such as `Set OPENAI_API_KEY in the environment before running this
stage` means the variable is not visible to the already-running notebook
kernel. Export it before launching Jupyter, then restart the kernel.

### Tavily SSL certificate failure

The notebook's currently saved output stops at the news stage with
`CERTIFICATE_VERIFY_FAILED`. This indicates that the Python environment could
not build a trusted certificate chain to `api.tavily.com`; it is not a scoring
logic error. Configure the correct organization/system CA bundle for the
environment and rerun the news cell. Do not work around this by disabling TLS
verification.

### Workbook schema errors

`pandas` raises an error if a required sheet or column was renamed or removed.
The notebook also intentionally fails if a contact's company has no matching
row in `Accounts Overview`. Correct the workbook data before continuing.

### Empty news response

The retrieval stage raises an explicit error when Tavily returns no dated
results. Verify the API key, network access, selected date window, and generated
query table before rerunning.

### Incomplete outputs

Later cells depend on objects created earlier. If the kernel was restarted or a
network stage failed, run the notebook from the top. Valid caches will be reused
automatically.

## Interpretation and limitations

- Search coverage depends on Tavily's index, the query construction, and the
  30-day window; absence from the candidate set cannot be repaired by ranking.
- The source-quality mapping is a small manually configured list. Unlisted
  domains receive a neutral value of 0.50, not a verified trust assessment.
- Entity detection is exact phrase matching for the company and competitors;
  aliases and subsidiaries may be missed.
- Keyword matching is a lightweight token-overlap baseline, not a production
  lexical retrieval system such as BM25.
- Explanations describe the scoring evidence but are not independent factual
  verification of the article.
- Reranking is limited to the top 20 hybrid candidates, so Cohere cannot recover
  an article excluded by the hybrid shortlist.
- Model and search API behavior can change over time. Record models, settings,
  run date, and reviewer protocol when reporting results.
- Human review is the decision criterion. Do not infer that a higher internal
  model score means a better recommendation without blinded evaluation.
