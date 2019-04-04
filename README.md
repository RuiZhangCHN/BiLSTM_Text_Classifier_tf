# BiLSTM_Text_Classifier_tf 基于BiLSTM的文本分类器

> A simple BiLSTM-based Text Classifier Implementation with Tensorflow v1.13+, which achieve about 0.973 accuracy rate on yelp review dataset.
> 
> 一个简单的基于Tensorflow v1.13+实现的Bi-LSTM文本分类器模型。该模型在Yelp评论数据集上的分类准确率达到约0.973。

### 1. 模型使用
#### i) 训练模型
首先根据需要修改main.py文件中flags参数设置。将mode修改为train，运行：

    python main.py
    
#### ii) 测试模型
修改mode为test，运行：

    python main,py

#### iii) 使用模型
需要自行实现solver.Solver.run()函数，注意修改batch_size=1。

### 2. 主要函数说明

#### i) prepro
 - **create_vocabulary:** 生成word2idx和idx2word。（可针对自己的数据集仿写该代码）
 - **create_yelp_ids:** 读取yelp数据文件生成ids文件。（可针对自己的数据集仿写该代码）

#### ii) model.BiLSTM:
 - **build_model:** 构建具有带训练参数的模型节点。
 - **build_graph:** 构建计算图。
 
#### iii) solver.Solver:
 - **load_data:** 从ids文件中读取数据
 - **prepare_text_batch:** 将长短不一的输入文本padding为相同长度的输入。
 - **train:** 使用train数据集训练模型，根据模型在dev集上的表现保存最优模型。
 - **test:** 测试模型在test数据集上的准确率。
 - **run:** \[Not Implemented\] 未实现的方法，用于使用训练好的模型进行预测。
 
### 3. 模型对比

| Model | Accuracy(Yelp) | Code |
| --- | ---| --- |
| **BiLSTM** | 0.973| [github](https://github.com/muenn/BiLSTM_Text_Classifier_tf) |
| CNN | - | - |
