python 11.7
cuda 12.4
根据官网提示，安装对应pytorch

数据集引用
@misc{
        title={瓶装白酒疵品检测数据集}
        url={https://tianchi.aliyun.com/dataset/dataDetail?dataId=110147}
        author={Tianchi},
        year={2021}
}


数据集分析:
1.3000*4096为标签图,492*658为瓶盖图
2.一张标签图  可以对应3种类型标签瑕疵

如果直接分类可能准确率较低,故对数据集图片按大小先行分类,

再训练多个卷积模型进行二分类,即判断图片是否存在该类型瑕疵,最后将结果整合

A数据集
图片,类B

B数据集训练时
图片,6瑕疵有无
图片,7瑕疵有无
图片,8瑕疵有无
