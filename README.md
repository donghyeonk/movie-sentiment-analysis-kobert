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
* Set nsmc_home_dir in main() to your nsmc dir

* Run
    ```
    cd KoBERT
    python3 movie_sentiment_analysis.py
    ```

* Performance
    * Accuracy 89.0% [(KoBERT fine-tuning: 90.1%)](https://github.com/SKTBrain/KoBERT#fine-tuning-performances)