# sjtu_ee228_M3DV

 EE228-Machine Learning（AI）cource project

## 环境

python3.6 pytorch0.4

## 文件

train.ipynb：训练

test.py：测试

model0.72316：模型参数，测试集（Leaderboard）上可达0.72316AUC

sampleSubmission.csv：Kaggle比赛提供的参考提交格式

## 代码说明

1. 新建test文件夹，将测试数据放入

2. 运行test.py文件可以生成test.csv的测试结果

3. 请保证test.py、model0.72316、sampleSubmission.csv在同一目录下，否则请自行修改test.py中的路径

4. data_path = './'(sampleSubmission.csv路径)
   
   model_path='model0.72316'（模型参数路径）（[下载链接](https://jbox.sjtu.edu.cn/l/noXqUh)）
   
   testpath='test/'(测试集数据路径)



## Kaggle比赛网址

[M3DV: Medical 3D Voxel Classification | Kaggle](https://www.kaggle.com/c/sjtu-ee228-2020)
