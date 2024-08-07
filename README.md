# Legal Clause Classification
RNN based Multi Class Classification of legal clauses


### Why do we need a Legal Clause Classifier (LCC)?

*Short Answer: To automate the Document review process and create a Legal AI Engine*

This repo contains the source code for api created using Starlette and the app was hosted by GCP App Engine for testing model outputs on real-world data. 

<!---[Test the API here](https://pensieve-gcp.appspot.com/)--->

## Background:
Recent developments in the field of Natural Language Inferencing (NLI), Understanding (NLU) and Generation (NLG) have opened potential opportunities to explore plethora or rich text information generated by domain experts for past several decades and made digitally accessible.

Legal domain is one such field where legal experts rely on the legal interpretation of text and also refer to similar precedents in legal documents. Hence, a data driven approach has potential to augment document discovery and information retrieval process by providing context aware outputs. While, fully automated legal services are far off in future, there are plenty of repetitive and mundane tasks, when automated can provide a boost to legal workflow.

In this work, I have focused on legal tasks related to contract abstraction and review process. In this paper, I have applied a Bi-directional LSTM neural network to classify the various legal clauses in order to augment the contract review process. While, baseline models (Random Forrest) with hand-generated features were able to achieve ~80% accuracy they still suffered with model generalization issues, when the input clause text is shorter or a sentence by sentence prediction is required. On the other hand, flexibility of tuning Bias and Variance together allowed an LSTM based Neural Network to reach ~95% accuracy on validation sets with much better model generalization.

## Model:
In this work, first a transfer learning based “Legal Language Model” was fine-tuned on Legal Corpora and the encoder from this model was subsequently trained for classification task. Such language model training provides inapplicable text generation; however training for this task provides enriched representation of legal vocabulary for further downstream tasks. Moreover, in this paper we provide evidence that a multi-layer representation extracted from this fine-tuned encoder is much more efficient (than pre-trained word vectors) for automating the process of culling out the specifics of a contract including relevant dates, clauses and pertinent information regarding parties, which are necessary for contract abstraction and helps automate part of document management lifecycle.

## Accuracy

- Baseline: Random Forrest 80%
- Deep Learning Model: AWD-LSTM Architecture 95%


While the training was carried out over a large labelled data set of approximately 1000 clauses for each category, 1% of this dataset was randomly selected for validation test. Following classification matrix shows the prediction power (~95% accuracy) of the RNN model.

### Validation Set
![](./RNN-AWD_LSTM_training.png)

## Inference:

This work showcases one of the projects towards building a legal AI engine for document intelligence.  Extracting document intelligence requires statistical learning from multiple documents but during inference we need to understand document from a macro perspective before getting into extracting variable and fixed elements of documents. We utilize a customized tokenizer built on top of SpaCy to recognize legal terminology followed by identifying Document Structure. While document text is easier to convert to text only, and discard formatting, taking the rich formatting into account provides much richer context behind the word.


## Integration in final product
For a demo of the product driven by this model please visit [Smriti.ai](https://www.smriti.ai)

## Reference:
App template for Google Compute Engine deployment was forked from the repo mentioned in following source.

Source: https://course.fast.ai/deployment_google_app_engine.html
