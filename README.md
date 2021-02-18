# Noise Eliminator for Active Learning (NEAL)
An Interactive Machine Learning based algorithm that helps produce automated systems that provide highly accurate predictions, with a low number of training instances .

**USE:** Can be used to improve predictions of any automated system and needs only a small subset of labelled data to produce stellar results.

**EXAMPLE:** Let's take the case of a ML system that analyzes CT scans to identify if a patient has lung cancer or not. It has a small pool of labeled data and a large pool of unlabeled data. The system finds patterns in the unlabeled data, assigns it a label and adds it to the training pool. If it encounters a data point whose label it's unsure of, it queries an oracle (a human or an expert system), in this case a doctor, to find the label for the data point. This is termed Active Learning.
  Sometimes the human expert might give it the wrong label, which is bad because if a person who does have a malignant tumor is classified as someone having a benign tumor, it has negative effects. NEAL is an algorithm that improves despite the wrong labels provided to it. 
