# Third-party notices

## BANKING77

Authors: Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson, Ivan Vulić.
Efficient Intent Detection with Dual Sentence Encoders, NLP for ConvAI, ACL 2020.
[Paper](https://arxiv.org/abs/2003.04807),
[original dataset](https://github.com/PolyAI-LDN/task-specific-datasets),
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/),
[repository license](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE).

Changes in this project: NFC/whitespace normalization for model input, casefold keys
for deduplication, removal of repeated/conflicting training records and train/test
text overlap, stratified training splits, derived metrics and error excerpts.
Raw CSV bytes and the official test are preserved. Reports containing source texts
and derived dataset excerpts retain this attribution and CC BY 4.0 terms. No author
endorsement is implied. Exact revisions/hashes are recorded in DATA_CARD and manifests.

## all-MiniLM-L6-v2

[Sentence Transformers model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache License 2.0 according to the model card.
[License text](https://www.apache.org/licenses/LICENSE-2.0).
Used as a frozen CPU encoder, without modifying pretrained weights. Weights are
downloaded to ignored local storage, not committed with project source.

## Software dependencies

Python dependencies retain their own licenses; exact packages are listed in uv.lock
and installed distribution metadata. The license of project code does not replace
dataset, model, or dependency licenses.
