"""Each loader on a tiny hand-made file in the dataset's official format."""
import csv
import gzip
import json

from rdgen.data import load


def test_mrqa(tmp_path):
    ctx = "Paris is in France. The capital of France is Paris. It is large."
    rows = [{"header": {"dataset": "SQuAD"}},
            {"context": ctx, "qas": [{"qid": "q1", "question": "What is the capital of France?",
                                      "answers": ["Paris"],
                                      "detected_answers": [{"text": "Paris", "char_spans": [[45, 49]]}]}]}]
    for name in ("SQuAD", "BioASQ"):
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    items = load("mrqa", in_domain=[str(tmp_path / "SQuAD.jsonl.gz")],
                 out_domain=[str(tmp_path / "BioASQ.jsonl.gz")])
    assert len(items) == 2 and items[0].evidence == [1]
    assert {i.split for i in items} >= {"test_shift"}


def test_qasper_with_pubmedqa_shift(tmp_path):
    paper = {"p1": {"abstract": "We study X.", "full_text": [{"section_name": "Intro", "paragraphs": [
        "We use BERT for the task.", "No ablation was run."]}],
        "qas": [
            {"question": "Do they use BERT?", "question_id": "a",
             "answers": [{"answer": {"unanswerable": False, "yes_no": True, "extractive_spans": [],
                                     "free_form_answer": "", "evidence": ["We use BERT for the task."]}}]},
            {"question": "Which dataset?", "question_id": "b",
             "answers": [{"answer": {"unanswerable": True, "yes_no": None, "extractive_spans": [],
                                     "free_form_answer": "", "evidence": []}}]}]}}
    for s in ("train", "dev", "test"):
        (tmp_path / f"{s}.json").write_text(json.dumps(paper))
    pq = {"123": {"QUESTION": "Does X help?", "CONTEXTS": ["X was given. Outcomes improved."],
                  "final_decision": "yes", "LABELS": ["A"], "LONG_ANSWER": "Yes."}}
    (tmp_path / "pq.json").write_text(json.dumps(pq))
    shift = load("pubmedqa", path=str(tmp_path / "pq.json"))
    items = load("qasper", train=str(tmp_path / "train.json"), valid=str(tmp_path / "dev.json"),
                 test=str(tmp_path / "test.json"), shift_items=shift)
    yes = [i for i in items if i.answer == "yes" and i.domain == "qasper"][0]
    assert yes.polarity == "pos" and yes.evidence == [1]
    assert any(i.polarity == "unanswerable" for i in items)
    assert any(i.split == "test_shift" for i in items)


def test_contractnli(tmp_path):
    text = "Recipient shall keep information secret. Obligations survive termination."
    doc = {"id": 1, "file_name": "0001234567-20-000001.htm", "text": text,
           "spans": [[0, 40], [41, len(text)]],
           "annotation_sets": [{"annotations": {
               "nda-1": {"choice": "Entailment", "spans": [1]},
               "nda-2": {"choice": "NotMentioned", "spans": []}}}]}
    labels = {"nda-1": {"hypothesis": "Some obligations survive termination."},
              "nda-2": {"hypothesis": "The agreement covers employees."}}
    (tmp_path / "c.json").write_text(json.dumps({"documents": [doc], "labels": labels}))
    items = load("contractnli", files=[str(tmp_path / "c.json")])
    pols = {i.meta["hypothesis_key"]: i.polarity for i in items}
    assert pols == {"nda-1": "pos", "nda-2": "unanswerable"}


def test_i2b2_2010(tmp_path):
    for site, assertion in (("partners", "absent"), ("beth", "present")):
        (tmp_path / site / "txt").mkdir(parents=True)
        (tmp_path / site / "ast").mkdir(parents=True)
        (tmp_path / site / "txt" / "r1.txt").write_text("Admission note .\nNo chest pain reported .\n")
        (tmp_path / site / "ast" / "r1.ast").write_text(
            f'c="chest pain" 2:1 2:2||t="problem"||a="{assertion}"\n')
    items = load("i2b2_2010", root=str(tmp_path))
    by_site = {i.meta["site"]: i for i in items}
    assert by_site["partners"].answer == "no" and by_site["partners"].polarity == "explicit_neg"
    assert by_site["beth"].split == "test_shift" and by_site["beth"].evidence == [1]


def test_bioasq(tmp_path):
    data = {"questions": [
        {"id": "q1", "type": "yesno", "body": "Is A linked to B?", "exact_answer": "yes",
         "snippets": [{"text": "A is linked to B."}]},
        {"id": "q2", "type": "yesno", "body": "Does C cause D?", "exact_answer": "no",
         "snippets": [{"text": "C does not cause D."}]}]}
    (tmp_path / "b.json").write_text(json.dumps(data))
    items = load("bioasq", path=str(tmp_path / "b.json"), n_distractors=1)
    assert all(len(i.evidence) == 1 for i in items)
    assert items[0].units[items[0].evidence[0]] == "A is linked to B."


def test_medqaro_polarity_file(tmp_path):
    rows = [{"patient_id": "1", "epicrisis": "Pacienta are cancer mamar. Fara metastaze osoase.",
             "question": "Metastaze osoase?", "answer": "Nu", "center": "Coltea", "cancer": "breast"},
            {"patient_id": "2", "epicrisis": "Pacient cu melanom.", "question": "Metastaze osoase?",
             "answer": "Nu", "center": "Dan Furtuna", "cancer": "melanoma"}]
    with (tmp_path / "m.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (tmp_path / "pol.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "polarity", "evidence"])
        w.writerow(["medqaro-1-0", "explicit_neg", "1"])
    items = load("medqaro", path=str(tmp_path / "m.csv"), polarity_file=str(tmp_path / "pol.csv"))
    by = {i.item_id: i for i in items}
    assert by["medqaro-1-0"].polarity == "explicit_neg" and by["medqaro-1-0"].evidence == [1]
    assert by["medqaro-2-1"].polarity == "unknown_neg" and by["medqaro-2-1"].split == "test_shift"
