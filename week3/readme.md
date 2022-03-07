# L1: Initial run without any optimizations:

Read 0M words
Number of words:  11144
Number of labels: 1340
Progress: 100.0% words/sec/thread:   14121 lr:  0.000000 avg.loss: 13.415264 ETA:   0h 0m 0s
(('__label__pcmcat174700050005',), array([0.00697159]))
N       9636
P@1     0.121

# L1: Changes
- Implemented title cleaning, minCount (used value = 100)
- model = fasttext.train_supervised(input="products.train.3", lr=1.0, epoch=100, wordNgrams=2)

# L1: Final run after changes:
(search_with_ml_week3) gitpod /workspace/search_with_machine_learning_course/week3 $ python fastTextTutorial.py 
Read 0M words
Number of words:  8132
Number of labels: 262
Progress: 100.0% words/sec/thread:   42133 lr:  0.000000 avg.loss:  0.079679 ETA:   0h 0m 0s
(('__label__pcmcat177200050011',), array([0.21443005]))
N       10000
P@1     0.812

# L2: Init run without any optimizations
Query word? iphone
Motorola 0.986259
Phone 0.976948
Phones 0.974025
Earphones 0.962868
Mobile 0.960405
Cables 0.957037
iPhone 0.954972
Bluetooth 0.953825
Bone 0.943132
Microphone 0.942807

# L2: Changes
Implemented name cleaning
~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/title_model -minCount 50 -epoch 25
sample threshold: 0.8

# L2: After all changes

Query word? iphone
4s 0.80914
apple 0.781616
ipod 0.737985
3gs 0.714567
ipad 0.665519
4thgeneration 0.637414
3g 0.570207
4 0.531041
touch 0.529277
shell 0.519676