可以解决留出验证在可用数据很少时，无法在统计学上代表数据的问题

##### 方法

将数据划分为大小相同的 K 个分区。对于每个分区 i，在剩余的 K-1 个分区上训练模型，然后在分区 i 上评估模型。最终分数等于 K 个分数的平均值。

![1554801470375](C:\Users\26910\AppData\Roaming\Typora\typora-user-images\1554801470375.png)

##### 代码

```python
k = 4
#验证集长度为分区长度（即原数据集长度/k）
num_validation_samples = len(data) // k

np.random.shuffle(data)

validation_scores = [] 
for fold in range(k):
   	#选择验证数据分区
     validation_data = data[num_validation_samples * fold: 
     num_validation_samples * (fold + 1)]
     #使用剩余数据作为训练数据。注意：+运算符是列表合并，不是求和
     training_data = data[:num_validation_samples * fold] + 
     data[num_validation_samples * (fold + 1):] 
    
    #创建一个全新（未训练过的）的模型实例
     model = get_model() 
     model.train(training_data)
     validation_score = model.evaluate(validation_data)
     validation_scores.append(validation_score)
    
#最终验证分数是k折验证分数的平均值
validation_score = np.average(validation_scores) 

#在所有非测试数据上训练最终模型
model = get_model() 
model.train(data)
test_score = model.evaluate(test_data)
```

