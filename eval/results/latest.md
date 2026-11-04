# Evaluation results

- Mode: **inproc**  |  LLM: `qwen2.5:3b-instruct`  |  Embeddings: `nomic-embed-text`  |  Reranker: `lexical`
- Run at: 2026-11-02 10:51:34

| Metric | Value | Target |
|---|---|---|
| pass_rate | 0.889 | >= 0.80 |
| groundedness | 1.0 | >= 0.90 |
| answer_relevance | 0.782 | stemmed token overlap diagnostic |
| context_relevance | 0.329 | stemmed token overlap diagnostic |
| coherence | 0.4 | LLM judge, 0-1; no release threshold |
| citation_accuracy | 1.0 | >= 0.90 |
| citation_coverage | 0.833 | tracked separately |
| out_of_doc_refusal | 1.0 | 1.00 |
| hallucinated_out_of_doc | 0 | 0 |
| hallucination_rate | 0.0 | 0.00 |
| unsupported_claims | 0 | 0 |
| recall_at_5 | 1.0 |  |
| recall_at_10 | 1.0 | >= 0.90 |
| mrr | 1.0 |  |
| latency_p50_ms | 17008.0 |  |
| latency_p95_ms | 46807.6 |  |

## Per question

| ID | Category | Pass | Grounded | Answer rel. | Context rel. | Citation acc. | Hallucination | Retrieval similarity | Coherence | Citations | Confidence | ms |
|---|---|---|---|---:|---:|---|---|---:|---:|---|---|---:|
| D1 | direct | PASS | True | 0.78 | 0.33 | True | False | 0.88 | 0.50 | Northwind_Leave_Policy.pdf p.1 | high | 22311 |
| D2 | direct | PASS | True | 0.60 | 1.00 | True | False | 0.79 | 0.50 | Northwind_Leave_Policy.pdf p.2 | high | 11369 |
| D3 | direct | PASS | True | 0.75 | 0.75 | True | False | 0.83 | 0.50 | Northwind_Leave_Policy.pdf p.2 | high | 11705 |
| D4 | direct | PASS | True | 1.00 | 0.43 | True | False | 0.73 | 0.00 | Northwind_Travel_Expense_Policy.pdf p.1 | high | 10894 |
| D5 | direct | PASS | True | 0.89 | 0.89 | True | False | 0.90 | 0.50 | Northwind_Travel_Expense_Policy.pdf p.2 | high | 9659 |
| D6 | direct | PASS | True | 0.75 | 0.75 | True | False | 0.76 | 0.00 | Northwind_Travel_Expense_Policy.pdf p.2 | high | 7562 |
| M1 | multi_section | PASS | True | 1.00 | 0.50 | True | False | 0.83 | 0.50 | Northwind_Leave_Policy.pdf p.1, Northwind_Leave_Policy.pdf p.4 | high | 37306 |
| M2 | multi_section | PASS | True | 0.86 | 0.71 | True | False | 0.82 | 0.50 | Northwind_Leave_Policy.pdf p.2, Northwind_Leave_Policy.pdf p.4 | high | 36145 |
| M3 | multi_section | FAIL | None | - | 0.00 | None | None | - | - | - | none | 28566 |
| M4 | multi_section | PASS | True | 0.70 | 0.30 | True | False | 0.82 | 0.50 | Northwind_Leave_Policy.pdf p.1, Northwind_Leave_Policy.pdf p.2 | high | 51703 |
| O1 | out_of_document | PASS | None | - | 0.00 | None | False | - | - | - | none | 2154 |
| O2 | out_of_document | PASS | None | - | 0.00 | None | False | - | - | - | none | 8856 |
| O3 | out_of_document | PASS | None | - | 0.00 | None | False | - | - | - | none | 22257 |
| A1 | ambiguous | PASS | True | 0.50 | 0.25 | None | False | 0.75 | 0.50 | Northwind_Leave_Policy.pdf p.1, Northwind_Leave_Policy.pdf p.2 | high | 40824 |
| A2 | ambiguous | PASS | None | - | 0.00 | None | None | - | - | - | none | 8043 |
| I1 | incorrect_premise | FAIL | None | - | 0.00 | None | None | - | - | - | none | 36781 |
| I2 | irrelevant | PASS | None | - | 0.00 | None | False | - | - | - | none | 2195 |
| I3 | irrelevant | PASS | None | - | 0.00 | None | False | - | - | - | none | 22037 |

## Answers

**D1** How many days of annual leave do full-time employees get per year?
> Every full-time employee is entitled to 24 paid days of annual leave per calendar year. [1]

**D2** When is a medical certificate required for sick leave?
> A medical certificate is required when the absence lasts more than 2 consecutive days. [1]

**D3** How many weeks is maternity leave?
> Maternity leave is 26 weeks, paid at 100 percent of base salary. [1]

**D4** What is the nightly hotel limit for Tier 1 cities such as London?
> [1] The nightly hotel limit for Tier 1 cities such as London is 250 USD.

**D5** Within how many days after a trip must the expense report be submitted?
> Expense reports must be submitted within 30 days after the trip ends. [1]

**D6** At what amount is a receipt required for an expense?
> A receipt is required for every expense above USD 25. [1]

**M1** What is the carry-forward limit for annual leave and by when must carried-forward days be used?
> The carry-forward limit for annual leave is 10 unused days into the next year. Carried-forward days must be used by 31 March, after which they expire without compensation. [1][2]

**M2** How many paid sick days and how many paid casual days do employees get?
> Employees get 12 paid sick days per year and 6 paid casual days per year.

**M3** Who must approve a trip costing USD 12,000, and what is the meal allowance for international trips?
> Information not available in the uploaded documents.

**M4** How much notice must be given before parental leave starts, and how long is paternity leave?
> The notice required before parental leave starts is 8 weeks, as stated in the document: "Employees must notify the People Team at least 8 weeks before the expected start date, except in cases of early birth or placement."

Paternity leave is 4 weeks, as stated in the document: "Paternity leave is 4 weeks, paid at 100 percent of base salary and may be taken within 6 months of the birth."

**O1** What is the company's stock price today?
> Information not available in the uploaded documents.

**O2** Who is the CEO of Northwind Labs?
> Information not available in the uploaded documents.

**O3** What is the dress code in the office?
> Information not available in the uploaded documents.

**A1** How many days off do I get?
> Based on the information provided in the documents, full-time employees are entitled to 24 paid days of annual leave per calendar year. There is no specific mention of the number of days off for sick leave or other types of leave.

**A2** What is the limit?
> Information not available in the uploaded documents.

**I1** Why does the policy give employees 30 days of annual leave?
> Information not available in the uploaded documents.

**I2** What is the weather in Delhi today?
> Information not available in the uploaded documents.

**I3** Write me a poem about hotels.
> Information not available in the uploaded documents.
