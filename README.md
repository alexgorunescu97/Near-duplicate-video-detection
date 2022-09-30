# Near-Duplicate Video Retrieval <br> with Deep Metric Learning
This project is inspired from the paper 
[Near-Duplicate Video Retrieval with Deep Metric Learning](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w5/Kordopatis-Zilos_Near-Duplicate_Video_Retrieval_ICCV_2017_paper.pdf). 
It provides code for training and evalutation of a Deep Metric Learning (DML) network on the problem of Near-Duplicate 
Video Retrieval (NDVR). During training, the DML network is fed with video triplets, generated by a *triplet generator*.
The network is trained based on the *triplet loss function*. 
For evaluation, *mean Average Precision* (*mAP*) and *Presicion-Recall curve* (*PR-curve*) are calculated.
Three publicly available dataset are supported, namely [VCDB](http://www.yugangjiang.info/research/VCDB/index.html), [CC_WEB_VIDEO](http://vireo.cs.cityu.edu.hk/webvideo/) and [FIVR-500k](https://ndd.iti.gr/fivr/).
