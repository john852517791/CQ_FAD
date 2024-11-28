# CQ_FAD

A Noval Feature via Color Quantisation for Fake Audio Detection [iscslp2024](https://arxiv.org/pdf/2408.10849)

In brief, this paper uses a color quantization reconstruction framework as a feature extractor for fake audio detection.

To truly understand the motivation behind this paper, readers need to be aware of a key fact: in the field of FAD, highly generalized models are those that use reconstruction-based pretrained models as feature extractors. (Of course, there have been highly effective models with very few parameters in this field, but unfortunately, those models have not been open-sourced.)

The aforementioned pretraining methods are all based on speech reconstruction, whether reconstructing waveforms or filter banks. Setting aside the factor of data volume, several key questions arise:

1. Is reconstruction as a method truly effective? If so, would a different reconstruction approach still be effective?
2. If reconstruction tasks are effective, would pretraining on real data beforehand further enhance FAD performance?
3. The features extracted by previous pretrained models have shown good results across various classifiers. Does this characteristic still hold true?

This paper attempts to address all the above questions at once. However, regrettably, it does not provide satisfactory answers, as the experimental results are not ideal.

I believe that exploring these questions is valuable, and I hope that more interpretable research about FAD will emerge in the future.





