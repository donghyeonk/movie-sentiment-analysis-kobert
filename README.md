# movie-sentiment-analysis-kobert

* Main components
    * [KoBERT of SKTBrain](https://github.com/SKTBrain/KoBERT)
    * [PyTorch](https://pytorch.org/)
    * [pytorch-transformers of huggingface](https://github.com/huggingface/pytorch-transformers)

* Install KoBERT
https://github.com/SKTBrain/KoBERT#how-to-install

* Get [Naver sentiment movie corpus](https://github.com/e9t/nsmc)
    ```
    git clone https://github.com/e9t/nsmc.git
    ```
* Set __nsmc_home_dir__ in __main()__ of __movie_sentiment_analysis.py__ to your nsmc dir

* Run
    ```
    cd movie-sentiment-analysis-kobert
    python3 movie_sentiment_analysis.py
    ```
    * It takes 36 minutes for each training epoch on an NVIDIA GeForce GTX 1070.

* Performance
    * Accuracy 89.55% (5th epoch) [(KoBERT fine-tuning: 90.1%)](https://github.com/SKTBrain/KoBERT#fine-tuning-performances)
