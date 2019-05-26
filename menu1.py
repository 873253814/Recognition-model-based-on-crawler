#!usr/bin/env python
#-*- coding:utf-8 -*-
from skimage import io,transform
import numpy as np
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt
import input_data 
import numpy as np
import model
import os 
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore,QtGui,QtWidgets
#from selenium import webdriver
import time
import debug
import ff
class filedialogdemo(QWidget):
    def __init__(self, parent=None):
        super(filedialogdemo, self).__init__(parent)
        layout = QVBoxLayout()

        self.content = QTextEdit()
        layout.addWidget(self.content)

        self.btn = QPushButton()
        self.btn.setText("爬取数据")
        self.btn.clicked.connect(self.buttonClicked)

        self.content2 = QTextEdit()
        layout.addWidget(self.content2)
        self.btn1 = QPushButton()
        self.btn1.setText("训练数据")
        self.btn1.clicked.connect(self.buttonClicked1)

        layout.addWidget(self.btn)
        layout.addWidget(self.btn1)

        self.btn = QPushButton()
        self.btn.clicked.connect(self.loadFile)
        self.btn.setText("从文件中获取照片")
        layout.addWidget(self.btn)

        self.label = QLabel()
        layout.addWidget(self.label)

        self.content1 = QTextEdit()
        layout.addWidget(self.content1)
        self.setWindowTitle("Identity")
        self.setLayout(layout)

    def buttonClicked1(self,keywords):
        self.content2.setPlainText(self.content2.toPlainText())
        string = self.content2.toPlainText()
        file_write_obj = open(r'test.txt', 'a+') # 以写的方式打开文件，如果文件不存在，就会自动创建
        count = len(open(r'test.txt','rU').readlines())
        file_write_obj.writelines(str(count)+":"+string)
        file_write_obj.write('\n')
        file_write_obj.close()
        ff.main()
    def buttonClicked(self,keywords):
        self.content.setPlainText(self.content.toPlainText())
        string = self.content.toPlainText()
        debug.main(string)

    def loadFile(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', r'E:\f\flower_photos', 'Image files(*.jpg *.gif *.png)')
        self.label.setPixmap(QPixmap(fname))
        path = os.getcwd()
        image = Image.open(fname)
    # 获取图片路径集和标签集
    # train, train_label = input_data.get_files(train_dir) 
        #driver=webdriver.Firefox()
        #driver.maximize_window()
        #driver.implicitly_wait(8)
        flower_dict={}

        with open('test.txt', 'r') as dict_file:
            for line in dict_file:
                (key, value) = line.strip().split(':')
                flower_dict[int(key)] = value
        w=100
        h=100
        c=3
 
        def read_one_image(fname):
            img = io.imread(fname)
            img = transform.resize(img,(w,h))
            return np.asarray(img)
 
        with tf.Session() as sess:
            data = []
            data1 = read_one_image(fname)

            data.append(data1)

 
            saver = tf.train.import_meta_graph('E:/f/flowers/model.ckpt.meta')
            saver.restore(sess,tf.train.latest_checkpoint('E:/f/flowers'))

            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            feed_dict = {x:data}
 
            logits = graph.get_tensor_by_name("logits_eval:0")
            print(logits)
            classification_result = sess.run(logits,feed_dict)
            #打印出预测矩阵
            print(classification_result)
            #打印出预测矩阵每一行最大值的索引
            print(tf.argmax(classification_result,1).eval())
            index = tf.argmax(classification_result,1).eval()
            probability = classification_result[0,index]
            #根据索引通过字典对应花的分类
            output = []
            output = tf.argmax(classification_result,1).eval()
            if(probability<2.3):
                self.content1.setText("非类别中的物种")
            else:
                for i in range(len(output)):
                    print("第",i+1,"朵花预测:"+flower_dict[output[i]])
                    self.content1.setText(flower_dict[output[i]])
                    #driver.get("https://www.baidu.com")#打开百度首页
                    #driver.find_element_by_xpath("//*[@id='kw']").send_keys(flower_dict[output[i]])#找到输入框并且填入”selenium”
                    #driver.find_element_by_xpath("//*[@id='su']").click()#然后点击“百度一下”



if __name__ == '__main__':
    app = QApplication(sys.argv)
    fileload =  filedialogdemo()
    fileload.show()
    sys.exit(app.exec_())
    evaluate_one_image()

