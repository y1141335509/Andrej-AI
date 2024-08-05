from MLP import MLP
#  /usr/bin/pip3 install torch     

if __name__ == '__main__':
   
  """
    这里是程序的入口。我们将给出一个简单的 input, output数据 xs, ys，然后搭建一个 MLP
    （Multiple Layer Perceptron）并用xs和ys进行训练。然后对比 MLP的预测值和真实值。
  """

  x = [2.0, 3.0, -1.0]    # input data
  n = MLP(3, [4, 4, 1])   # 第一个hidden layer有 4 个neurons；第二个hidden layer有 4 个neurons；最后的输出层只有一个neuron。所以是binary-classification
  n(x)

  xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
  ]                             # input data
  ys = [1.0, -1.0, -1.0, 1.0]   # desired targets

  learning_rate = 0.1
  for k in range(20):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0          # 确保每次backward之前都将每一个 参数的 梯度值 归零
    loss.backward()

    # update parameters
    for p in n.parameters():
        p.data += -learning_rate * p.grad

    print(f'Loss at step {k}: {loss.data}', ypred)

  print(ypred)